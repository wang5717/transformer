# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Modifications copyright (C) 2024 Maksim Ploter
import torch

from .backbone import build_backbone
from .deformable_detr import DeformableDETR, DeformablePostProcess
from .detr import DETR, PostProcess, SetCriterion
from .detr_segmentation import (DeformableDETRSegm, DeformableDETRSegmTracking,
                                DETRSegm, DETRSegmTracking,
                                PostProcessPanoptic, PostProcessSegm)
from .detr_tracking import DeformableDETRTracking, DETRTracking
from .matcher import build_matcher
from .perceiver_detection import build_model as build_model_perceiver_detection
from .perceiver_tracking import PerceiverTracking
from .transformer import build_transformer


def build_model(args):
    if args.dataset == 'coco':
        num_classes = 91
    elif args.dataset == 'coco_panoptic':
        num_classes = 250
    elif args.dataset in ['coco_person', 'mot', 'mot_crowdhuman', 'crowdhuman', 'mot_coco_person']:
        # We inherited the 20 classes setup from the original Deformable DETR code.
        # The focal loss is more stable with more classes. You can train with any amount of output neurons.
        # It will just learn to never predict the others and only the one class or background.
        # https://github.com/timmeinhardt/trackformer/issues/120#issuecomment-1885045880

        # Computing the focal loss for a single class introduces some noise
        # which we found to be less if we increase the number of classes. The number 20 is a bit arbitrary here.
        # https://github.com/timmeinhardt/trackformer/issues/50#issuecomment-1173942985
        num_classes = 20
        if args.model == 'perceiver':
            # Not clear why TrackFormer reduces number of classes to 20
            # Revert it to number of classes in COCO dataset
            # Perceiver was trained on COCO

            # IN COCO_CLASSES[:2] -> [N/A, Person]
            # We are only interested in 1 clas Person,
            # so N is 1, but according to DETR paper we add 1 (account for zero indexed array)
            num_classes = 2
    else:
        raise NotImplementedError

    device = torch.device(args.device)
    matcher = build_matcher(args)

    if args.model == 'perceiver':
        model = build_model_perceiver_based(args, matcher, num_classes)
    else:
        model = build_model_detr_based(args, matcher, num_classes)

    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef,}

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses.append('masks')

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tracking=args.tracking,
        track_query_false_positive_eos_weight=args.track_query_false_positive_eos_weight,)
    criterion.to(device)

    if args.focal_loss:
        postprocessors = {'bbox': DeformablePostProcess()}
    else:
        postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors


def build_model_perceiver_based(args, matcher, num_classes):
    backbone, perceiver, classifier_head = build_model_perceiver_detection(args, matcher, num_classes)

    tracking_kwargs = {
        'track_query_false_positive_prob': args.track_query_false_positive_prob,
        'track_query_false_negative_prob': args.track_query_false_negative_prob,
        'matcher': matcher,
        'backprop_prev_frame': args.track_backprop_prev_frame,
    }

    detection_model_kwargs = {
        'backbone': backbone,
        'perceiver': perceiver,
        'classification_head': classifier_head,
    }

    model = PerceiverTracking(
        tracking_kwargs=tracking_kwargs,
        detection_model_kwargs=detection_model_kwargs
    )

    return model


def build_model_detr_based(args, matcher, num_classes):
    backbone = build_backbone(args)
    detr_kwargs = {
        'backbone': backbone,
        'num_classes': num_classes - 1 if args.focal_loss else num_classes,
        'num_queries': args.num_queries,
        'aux_loss': args.aux_loss,
        'overflow_boxes': args.overflow_boxes}
    tracking_kwargs = {
        'track_query_false_positive_prob': args.track_query_false_positive_prob,
        'track_query_false_negative_prob': args.track_query_false_negative_prob,
        'matcher': matcher,
        'backprop_prev_frame': args.track_backprop_prev_frame, }
    mask_kwargs = {
        'freeze_detr': args.freeze_detr}
    if args.deformable:
        raise NotImplementedError('Deformable transformer is not supported.')
    else:
        transformer = build_transformer(args)

        detr_kwargs['transformer'] = transformer

        if args.tracking:
            if args.masks:
                model = DETRSegmTracking(mask_kwargs, tracking_kwargs, detr_kwargs)
            else:
                model = DETRTracking(tracking_kwargs, detr_kwargs)
        else:
            if args.masks:
                model = DETRSegm(mask_kwargs, detr_kwargs)
            else:
                model = DETR(**detr_kwargs)
    return model
