# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmdet.models.dense_heads import \
        Mask2FormerHead as MMDET_Mask2FormerHead
except ModuleNotFoundError:
    MMDET_Mask2FormerHead = None

from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList
from mmcv.cnn import build_norm_layer, ConvModule

class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=[3, 4, 5, 6]):
        super().__init__()

        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch = in_channel // len(sizes)
        self.channel_splits = [ch, ch, ch, ch]
        self.out_size = sizes[-1]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        feats = torch.split(feats, self.channel_splits, dim=1)
        priors = [F.upsample(input=self.stages[i](feats[i]), size=(self.out_size, self.out_size),
                             mode='bilinear') for i in range(4)]
        bottle = torch.cat(priors, 1)

        return self.relu(bottle).flatten(2).permute(2, 0, 1)


class HighPassModule(nn.Module):
    def __init__(self, in_channel, sizes=[3, 5, 7, 9], pool_sizes=[5, 6, 7, 8]):
        super().__init__()

        ch = in_channel // len(sizes)
        self.channel_splits = [ch, ch, ch, ch]
        self.stages = nn.ModuleList()
        self.pools = nn.ModuleList()

        for size, pool_size in zip(sizes, pool_sizes):
            dilation = 1
            padding_size = (size + (size - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                ch,
                ch,
                kernel_size=(size, size),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=ch,
            )
            self.stages.append(cur_conv)
            cur_pool = nn.AdaptiveMaxPool2d((pool_size, pool_size))
            self.pools.append(cur_pool)

        self.out_size = pool_sizes[-1]
        self.proj = nn.Conv2d(in_channel, in_channel, 1)
        self.relu = nn.ReLU()

    def forward(self, feats):
        feats_list = torch.split(feats, self.channel_splits, dim=1)
        high_out = [conv(x) for conv, x in zip(self.stages, feats_list)]
        high_out = torch.cat(high_out, dim=1)
        feature = self.proj(feats) * high_out
        feature_list = torch.split(feature, self.channel_splits, dim=1)
        out_list = [F.upsample(input=self.pools[i](feature_list[i]),
                               size=(self.out_size, self.out_size),
                               mode='bilinear') for i in range(len(self.pools))]
        out = torch.cat(out_list, dim=1)

        return self.relu(out).flatten(2).permute(2, 0, 1)


@MODELS.register_module()
class MaskCCHead(MMDET_Mask2FormerHead):
    """Implements the MaskCCHead head.

    Args:
        num_classes (int): Number of classes. Default: 150.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        ignore_index (int): The label index to be ignored. Default: 255.
        low_pass_kernel (list):  AdaptiveAvgPool2d setting for low pass query in 
            Pooled Query Vector Generator. Default: [3, 4, 5, 6].
        high_pass_kernel (list): Depthwise conv setting for high pass query in 
            Pooled Query Vector Generator. Default: [3, 5, 7, 9].
        high_pool_kernel (list): AdaptiveMaxPool2d setting for high pass query in
            Pooled Query Vector Generator. Default: [5, 6, 7, 8].
        loss_contrast (ConfigType): MaskCC loss setting
        mask_ind (list): the ids for layers in Transformer Deocder used for 
            MaskCC loss. The 0th layer is not counted in my thesis, so the 
            num of default shallow layers is 3. Default: [0,1,2,3,9].
    """

    def __init__(self,
                 num_classes,
                 align_corners=False,
                 ignore_index=255,
                 low_pass_kernel=[3, 4, 5, 6],
                 high_pass_kernel=[3, 5, 7, 9],
                 high_pool_kernel=[5, 6, 7, 8],
                 loss_contrast=None,
                 mask_ind=[0,1,2,3,9],
                 **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index

        feat_channels = kwargs['feat_channels']
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)

        if low_pass_kernel is not None:
            self.low_pass_query_generator = LowPassModule(feat_channels, low_pass_kernel)
        else:
            self.low_pass_query_generator = None

        if high_pass_kernel is not None:
            self.high_pass_query_generator = HighPassModule(feat_channels, high_pass_kernel,
                                                        high_pool_kernel)
        else:
            self.high_pass_query_generator = None

        if low_pass_kernel is not None and high_pass_kernel is not None:
            self.proj = nn.Linear(feat_channels, feat_channels)
            self.query_norm = build_norm_layer(dict(type='SyncBN', requires_grad=True),
                                    feat_channels)[1]
        
        if loss_contrast is not None:
            self.loss_contrast = MODELS.build(loss_contrast)
            in_channels = kwargs['in_channels']
            self.contrast_proj = nn.Sequential(
                ConvModule(
                    feat_channels * len(in_channels), feat_channels,
                    kernel_size=1, padding=0, dilation=1,
                    conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='SyncBN', requires_grad=True),
                    act_cfg=dict(type='ReLU')),
                ConvModule(
                    feat_channels, feat_channels, kernel_size=1,
                    padding=0, dilation=1, conv_cfg=dict(type='Conv2d'),
                    norm_cfg=None, act_cfg=None))

            self.mask_ind = mask_ind
            self.contrast_mask_proj = nn.Sequential(
                ConvModule(
                    self.num_queries, feat_channels,
                    kernel_size=1, padding=0, dilation=1,
                    conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='SyncBN', requires_grad=True),
                    act_cfg=dict(type='ReLU')),
                ConvModule(
                    feat_channels, feat_channels, kernel_size=1,
                    padding=0, dilation=1, conv_cfg=dict(type='Conv2d'),
                    norm_cfg=None, act_cfg=None))
            
            

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_Mask2FormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long()

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)
        
        if hasattr(self, 'contrast_proj') is False:
            all_cls_scores, all_mask_preds = self(x, batch_data_samples)
            # loss
            losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                    batch_gt_instances, batch_img_metas)
            return losses
        
        all_cls_scores, all_mask_preds, proj_feats, proj_masks = self(x, batch_data_samples)
        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                batch_gt_instances, batch_img_metas)
        
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        groundtruth = torch.stack(gt_semantic_segs, dim=0)

        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        
        contrast_losses_dict = self.loss_contrast(proj_feats, seg_logits, 
                                groundtruth, proj_masks)
        for loss_name, loss_val in contrast_losses_dict.items():
            losses[loss_name] = loss_val
        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        batch_size = len(batch_img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        if self.low_pass_query_generator is not None:
            low_pass_query = self.low_pass_query_generator(mask_features)
        else:
            low_pass_query = None

        if self.high_pass_query_generator is not None:
            high_pass_query = self.high_pass_query_generator(mask_features)
        else:
            high_pass_query = None

        if low_pass_query is not None and high_pass_query is not None:
            if low_pass_query.shape[0] + high_pass_query.shape[0] == query_feat.shape[0]:
                generate_query = self.proj(torch.cat((low_pass_query,
                                            high_pass_query), dim=0))
                generate_query = torch.einsum('bcn->nbc', self.query_norm(
                                    torch.einsum('nbc->bcn', generate_query)))
                query_feat = query_feat + generate_query
            elif low_pass_query.shape[0] == query_feat.shape[0] and \
                    high_pass_query.shape[0] == query_feat.shape[0]:
                generate_query = self.proj(low_pass_query + high_pass_query)
                generate_query = torch.einsum('bcn->nbc', self.query_norm(
                                    torch.einsum('nbc->bcn', generate_query)))
                query_feat = query_feat + generate_query
            else:
                raise Exception(f"the shapes of low_pass_query ({low_pass_query.shape}) and "
                                f"high_pass_query ({high_pass_query.shape}) is not compatible with"
                                f"query_feat ({query_feat.shape})")
        elif low_pass_query is not None:
            if low_pass_query.shape[0] == query_feat.shape[0]:
                query_feat = query_feat + torch.einsum('bcn->nbc', self.query_norm(
                                    torch.einsum('nbc->bcn', self.proj(low_pass_query))))
            else:
                raise Exception(f"the shape of low_pass_query ({low_pass_query.shape}) is "
                                f"not compatible with query_feat ({query_feat.shape})")
        elif high_pass_query is not None:
            if high_pass_query.shape[0] == query_feat.shape[0]:
                query_feat = query_feat + torch.einsum('bcn->nbc', self.query_norm(
                                    torch.einsum('nbc->bcn', self.proj(high_pass_query))))
            else:
                raise Exception(f"the shape of high_pass_query ({high_pass_query.shape}) is "
                                f"not compatible with query_feat ({query_feat.shape})")

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        if hasattr(self, 'contrast_proj') and self.training:
            multi_scale_memorys.append(mask_features)
            upsampled_inputs = [
                F.interpolate(
                    input=x, size=multi_scale_memorys[-1].shape[2:],
                    mode='bilinear', align_corners=False
                ) for x in multi_scale_memorys
            ]
            concat_feats = torch.cat(upsampled_inputs, dim=1)
            proj_feats = self.contrast_proj(concat_feats)
            proj_masks = []
            for ind in self.mask_ind:
                proj_masks.append(self.contrast_mask_proj(mask_pred_list[ind]))
            return cls_pred_list, mask_pred_list, proj_feats, proj_masks
        
        return cls_pred_list, mask_pred_list