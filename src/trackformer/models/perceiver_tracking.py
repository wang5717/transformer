import torch

from trackformer.models.perceiver_detection import PerceiverDetection
from trackformer.util.misc import NestedTensor


class PerceiverTracking(PerceiverDetection):
    def __init__(self, tracking_kwargs, detection_model_kwargs):
        PerceiverDetection.__init__(self, **detection_model_kwargs)

    def forward(self, samples: NestedTensor, targets: list = None, **kwargs):
        src, mask = samples.decompose()

        if len(src.shape) < 5:
            # samples without a time dimension
            return super().forward(samples, targets)

        src = src.permute(1, 0, 2, 3, 4)  # change dimension order from BT___ to TB___

        result = {'pred_logits': [], 'pred_boxes': []}
        latents = None

        targets_flat = []
        for timestamp, batch in enumerate(src):
            current_targets = [target_list[timestamp] for target_list in targets]
            frame_keep_mask = [t['keep_frame'] for t in current_targets]
            frame_keep_mask = torch.tensor(frame_keep_mask, device=batch.device)
            frame_keep_mask = frame_keep_mask.view(-1, 1, 1, 1)
            batch = batch * frame_keep_mask
            out, *_ = super().forward(
                samples=batch, targets=current_targets, latents=latents
            )
            latents = out['hs_embed']
            result['pred_logits'].append(out['pred_logits'])
            result['pred_boxes'].append(out['pred_boxes'])
            targets_flat.extend(current_targets)

        result['pred_logits'] = torch.cat(result['pred_logits'], dim=0)
        result['pred_boxes'] = torch.cat(result['pred_boxes'], dim=0)
        return result, targets_flat
