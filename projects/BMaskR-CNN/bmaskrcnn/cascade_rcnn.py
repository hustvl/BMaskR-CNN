# Copyright (c) wondervictor. All Rights Reserved
import torch
from typing import Dict, List, Union

from detectron2.layers import ShapeSpec
from detectron2.structures import Instances

from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads
from detectron2.modeling.poolers import ROIPooler
from .roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals


@ROI_HEADS_REGISTRY.register()
class CascadeBoundaryROIHeads(CascadeROIHeads):

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)

    def _init_mask_head(self, cfg, input_shape):
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # edge poolers
        boundary_resolution     = cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_RESOLUTION
        boundary_in_features    = cfg.MODEL.BOUNDARY_MASK_HEAD.IN_FEATURES
        boundary_scales         = tuple(1.0 / input_shape[k].stride for k in boundary_in_features)
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        self.mask_in_features = in_features
        self.boundary_in_features = boundary_in_features
        # ret = {"mask_in_features": in_features}
        self.mask_pooler= ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.boundary_pooler = ROIPooler(
            output_size=boundary_resolution,
            scales=boundary_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        mask_features = [features[f] for f in self.mask_in_features]
        boundary_features = [features[f] for f in self.boundary_in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(mask_features, proposal_boxes)
            boundary_features = self.boundary_pooler(boundary_features, proposal_boxes)
            return self.mask_head(mask_features, boundary_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(mask_features, pred_boxes)
            boundary_features = self.boundary_pooler(boundary_features, pred_boxes)
            return self.mask_head(mask_features, boundary_features, instances)

