# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Modifications copyright (C) 2024 Maksim Ploter

import math
import random
from contextlib import nullcontext

import torch
import torch.nn as nn

from ..util import box_ops
from ..util.misc import NestedTensor, get_rank
from .deformable_detr import DeformableDETR
from .detr import DETR
from .matcher import HungarianMatcher


class DETRTrackingBase(nn.Module):

    def __init__(self,
                 track_query_false_positive_prob: float = 0.0,
                 track_query_false_negative_prob: float = 0.0,
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame=False):
        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._backprop_prev_frame = backprop_prev_frame

        self._tracking = False

    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        """Sets the module in tracking mode."""
        self.eval()
        self._tracking = True

    def add_track_queries_to_targets(self, targets, prev_indices, prev_out, add_false_pos=True):
        """
            Enhances ground-truth targets by adding track queries from the previous frame's outputs.

            This method manages both positive and negative track queries to assist in associating objects
            detected in consecutive frames. It is used during the training and evaluation of tracking models
            based on the DETR architecture.

            Args:
                targets (list[dict]):
                    A list of dictionaries where each dictionary contains the ground truth for a sample in the batch.
                    Each dictionary should include the current frame's object bounding boxes and labels.

                prev_indices (list[tuple]):
                    A list of tuples where each tuple contains:
                    - prev_out_ind (torch.Tensor): Indices of the previous frame's predictions that match with the current targets.
                    - prev_target_ind (torch.Tensor): Indices of the ground truth targets from the previous frame that match the predictions.

                prev_out (dict):
                    A dictionary containing the outputs from the previous frame, including:
                    - 'pred_boxes' (torch.Tensor): Predicted bounding boxes from the previous frame.
                    - 'hs_embed' (torch.Tensor): Embeddings from the previous frame.
                    Additional items may be included depending on the implementation.

                add_false_pos (bool, optional):
                    If True, adds false positive track queries to the targets by randomly selecting unmatched predictions
                    from the previous frame. This simulates incorrect predictions and helps the model learn to identify
                    and ignore false positives. Default is True.

            Returns:
                None:
                    The method modifies the `targets` list in place. It updates the targets with additional track queries,
                    their corresponding embeddings, bounding boxes, and masks.
        """

        device = prev_out['pred_boxes'].device

        # Calculate the minimum number of previous frame targets across all batches
        min_prev_target_ind = min([len(prev_ind[1]) for prev_ind in prev_indices])
        # Initialize the number of previous frame targets to zero
        num_prev_target_ind = 0
        # If there are any previous frame targets, randomly select a number of them to use
        if min_prev_target_ind:
            # Generate a random number between 0 and min_prev_target_ind (inclusive)
            # to determine how many targets to use
            num_prev_target_ind = torch.randint(0, min_prev_target_ind + 1, (1,)).item()

        # Initialize the number of false positive track queries to zero
        num_prev_target_ind_for_fps = 0
        # If there are previous frame targets, compute how many of them should be false positives
        if num_prev_target_ind:
            # Calculate the number of false positive track queries based on the false positive probability
            # Use math.ceil to ensure we round up to the nearest integer
            num_prev_target_ind_for_fps = \
                torch.randint(int(math.ceil(self._track_query_false_positive_prob * num_prev_target_ind)) + 1, (1,)).item()

        for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
            prev_out_ind, prev_target_ind = prev_ind

            # If the false negative probability is non-zero, randomly select a subset of previous targets
            if self._track_query_false_negative_prob:
                # Create a random permutation of indices to select a subset of previous targets
                # Use num_prev_target_ind to determine the size of the subset
                random_subset_mask = torch.randperm(len(prev_target_ind))[:num_prev_target_ind]

                # Apply the random subset mask to select a subset of previous output and target indices
                prev_out_ind = prev_out_ind[random_subset_mask]
                prev_out_ind = prev_out_ind.to(device)
                prev_target_ind = prev_target_ind[random_subset_mask]  # Select corresponding target indices

            # detected prev frame tracks
            # Extract track IDs from the previous frame for the selected target indices
            prev_track_ids = target['prev_target']['track_ids'][prev_target_ind]

            # match track ids between frames
            # Create a matrix to match previous frame track IDs with current frame track IDs
            # The matrix will have dimensions (number of previous track IDs) x (number of current track IDs)
            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])

            # Determine which previous track IDs have a match in the current frame
            # `any(dim=1)` checks if there's at least one match for each previous track ID
            target_ind_matching = target_ind_match_matrix.any(dim=1).to(device)

            # Get indices of matched current frame track IDs for each previous track ID
            target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

            # index of prev frame detection in current frame box list
            # Store indices of current frame track IDs that correspond to previous frame detections
            target['track_query_match_ids'] = target_ind_matched_idx

            # random false positives
            # Add random false positives to the predictions if specified
            if add_false_pos and self._track_query_false_positive_prob:
                # Extract the matched predicted boxes from the previous frame for the current targets
                prev_boxes_matched = prev_out['pred_boxes'][i, prev_out_ind[target_ind_matching]]

                # Create a list of indices for the predicted boxes that were not matched in the previous step
                not_prev_out_ind = torch.arange(prev_out['pred_boxes'].shape[1])
                not_prev_out_ind = [
                    ind.item()
                    for ind in not_prev_out_ind
                    if ind not in prev_out_ind]

                # List to hold indices of random false positives
                random_false_out_ind = []

                # Select a subset of previous target indices to generate false positives
                prev_target_ind_for_fps = torch.randperm(num_prev_target_ind)[:num_prev_target_ind_for_fps]

                # Iterate through the selected previous target indices for generating false positives
                for j in prev_target_ind_for_fps:
                    # Extract the unmatched predicted boxes from the previous frame
                    prev_boxes_unmatched = prev_out['pred_boxes'][i, not_prev_out_ind]

                    # Compute distances between matched boxes and unmatched boxes
                    if len(prev_boxes_matched) > j:
                        prev_box_matched = prev_boxes_matched[j]
                        box_weights = \
                            prev_box_matched.unsqueeze(dim=0)[:, :2] - \
                            prev_boxes_unmatched[:, :2]
                        box_weights = box_weights[:, 0] ** 2 + box_weights[:, 0] ** 2
                        box_weights = torch.sqrt(box_weights)

                        # Choose a random false positive based on the computed weights
                        random_false_out_idx = not_prev_out_ind.pop(
                            torch.multinomial(box_weights.cpu(), 1).item())
                    else:
                        # If there are not enough matched boxes, pick a random index from unmatched boxes
                        random_false_out_idx = not_prev_out_ind.pop(torch.randperm(len(not_prev_out_ind))[0])

                    # Add the selected false positive index to the list
                    random_false_out_ind.append(random_false_out_idx)

                prev_out_ind = torch.tensor(prev_out_ind.tolist() + random_false_out_ind).long()

                # Update the match matrix to include false positives
                target_ind_matching = torch.cat([
                    target_ind_matching,
                    torch.tensor([False, ] * len(random_false_out_ind)).bool().to(device)
                ])

            # Create masks for track queries
            # Mask indicating which track queries are active (matched) vs. inactive (not matched)
            track_queries_mask = torch.ones_like(target_ind_matching).bool()
            track_queries_fal_pos_mask = torch.zeros_like(target_ind_matching).bool()
            track_queries_fal_pos_mask[~target_ind_matching] = True

            # Put previous hidden state as it is for perceiver model
            target['_hs_embed'] = prev_out['hs_embed'][i]
            # Update target information with the previous frame's embeddings and bounding boxes
            target['track_query_hs_embeds'] = prev_out['hs_embed'][i, prev_out_ind]
            target['track_query_boxes'] = prev_out['pred_boxes'][i, prev_out_ind].detach()

            # Extend masks to include placeholders for additional queries if necessary
            target['track_queries_mask'] = torch.cat([
                track_queries_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

            target['track_queries_fal_pos_mask'] = torch.cat([
                track_queries_fal_pos_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        if targets is not None and not self._tracking:
            prev_targets = [target['prev_target'] for target in targets]

            # if self.training and random.uniform(0, 1) < 0.5:
            if self.training:
            # if True:
                backprop_context = torch.no_grad
                if self._backprop_prev_frame:
                    backprop_context = nullcontext

                with backprop_context():
                    if 'prev_prev_image' in targets[0]:
                        for target, prev_target in zip(targets, prev_targets):
                            prev_target['prev_target'] = target['prev_prev_target']

                        prev_prev_targets = [target['prev_prev_target'] for target in targets]

                        # PREV PREV
                        prev_prev_out, _, prev_prev_features, _, _ = super().forward([t['prev_prev_image'] for t in targets])

                        prev_prev_outputs_without_aux = {
                            k: v for k, v in prev_prev_out.items() if 'aux_outputs' not in k}
                        prev_prev_indices = self._matcher(prev_prev_outputs_without_aux, prev_prev_targets)

                        self.add_track_queries_to_targets(
                            prev_targets, prev_prev_indices, prev_prev_out, add_false_pos=False)

                        # PREV
                        prev_out, _, prev_features, _, _ = super().forward(
                            [t['prev_image'] for t in targets],
                            prev_targets,
                            prev_prev_features)
                    else:
                        prev_out, _, prev_features, _, _ = super().forward([t['prev_image'] for t in targets])

                    # prev_out = {k: v.detach() for k, v in prev_out.items() if torch.is_tensor(v)}

                    prev_outputs_without_aux = {
                        k: v for k, v in prev_out.items() if 'aux_outputs' not in k}
                    prev_indices = self._matcher(prev_outputs_without_aux, prev_targets)

                    self.add_track_queries_to_targets(targets, prev_indices, prev_out)
            else:
                # if not training we do not add track queries and evaluate detection performance only.
                # tracking performance is evaluated by the actual tracking evaluation.
                for target in targets:
                    device = target['boxes'].device

                    target['track_query_hs_embeds'] = torch.zeros(0, self.hidden_dim).float().to(device)
                    # target['track_queries_placeholder_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_queries_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_queries_fal_pos_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_query_boxes'] = torch.zeros(0, 4).to(device)
                    target['track_query_match_ids'] = torch.tensor([]).long().to(device)

        out, targets, features, memory, hs  = super().forward(samples, targets, prev_features)

        return out, targets, features, memory, hs


# TODO: with meta classes
class DETRTracking(DETRTrackingBase, DETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)


class DeformableDETRTracking(DETRTrackingBase, DeformableDETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)
