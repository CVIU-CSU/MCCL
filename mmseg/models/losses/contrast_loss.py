# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES

@LOSSES.register_module()
class ContrastLoss(nn.Module):
    def __init__(self,
                 max_sample=1024,
                 threshold=100,
                 feat_loss_weight=0.1,
                 mask_loss_weight=0.1,
                 temperature=0.1,
                 base_temperature=0.07,
                 ignore_index=255,
                 loss_name='loss_contrast'):
        super(ContrastLoss, self).__init__()
        self.max_sample = max_sample
        self.threshold = threshold
        self.feat_loss_weight = feat_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self._loss_name = loss_name
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.base_temperature = base_temperature

    def _hard_sample_mining(self, predict, label, feature_map, probability, proj_masks):
        B = feature_map.shape[0]
        hard_sample_classes = []
        total_classes = 0

        for i in range(B):
            pixels_label_per = label[i]
            superpixels_class = torch.unique(pixels_label_per)
            class_per = [x for x in superpixels_class if x != self.ignore_index]
            class_per = [x for x in class_per if (pixels_label_per == x).nonzero().shape[0] > self.threshold]

            hard_sample_classes.append(class_per)
            total_classes += len(class_per)

        if total_classes == 0:
            return None, None, None, None

        feature_list = []
        label_list = []
        mask_anchor_list = []
        mask_key_list = []
        
        anchor_mask = proj_masks[-1]
        key_mask = torch.cat([x.unsqueeze(2) for x in proj_masks[:-1]], dim=2)
            
        n_view = self.max_sample // total_classes
        n_view = min(n_view, self.threshold)

        for i in range(B):
            pixels_label_per = label[i]
            pixels_predict_per = predict[i]
            pixels_feature_per = feature_map[i]
            pixels_mask_anchor_per = anchor_mask[i]
            pixels_mask_key_per = key_mask[i]
            pixels_probability_per = probability[i]
            hard_sample_classes_per = hard_sample_classes[i]

            for cls_id in hard_sample_classes_per:
                hard_indices = ((pixels_label_per == cls_id) & (pixels_predict_per != cls_id)).nonzero().squeeze(1)
                easy_indices = ((pixels_label_per == cls_id) & (pixels_predict_per == cls_id)).nonzero().squeeze(1)

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    num_hard_keep = num_hard
                    num_easy_keep = num_easy

                cls_pixels_logits_per = pixels_probability_per[hard_indices, cls_id]
                cls_pixels_logits_per = cls_pixels_logits_per.argsort(dim=0, descending=True)
                hard_indices = hard_indices[cls_pixels_logits_per[:num_hard_keep]]
                
                cls_pixels_logits_per = pixels_probability_per[easy_indices, cls_id]
                cls_pixels_logits_per = cls_pixels_logits_per.argsort(dim=0)
                easy_indices = easy_indices[cls_pixels_logits_per[:num_easy_keep]]

                indices = torch.cat((hard_indices, easy_indices), dim=0)
                feature_list.append(pixels_feature_per[indices, :])
                label_list.append(pixels_label_per[indices])
                mask_anchor_list.append(pixels_mask_anchor_per[indices, :])
                mask_key_list.append(pixels_mask_key_per[indices, :, :])

        selected_feature = torch.cat(feature_list, dim=0)
        selected_label = torch.cat(label_list, dim=0)
        selected_mask_anchor = torch.cat(mask_anchor_list, dim=0)
        selected_mask_key = torch.cat(mask_key_list, dim=0)

        return selected_feature, selected_label, selected_mask_anchor, selected_mask_key

    def feat_contrastive(self, feature, label):

        label = label.unsqueeze(dim=1)
        label = torch.eq(label, label.T)

        size = feature.shape[0]
        dot_contrast = torch.div(
            torch.matmul(feature, feature.T),
            self.temperature)

        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        dot_contrast = dot_contrast - logits_max.detach()

        logits_mask = (~torch.eye(size, dtype=torch.bool)).to(label.device)
        label = torch.mul(label, logits_mask.detach())

        exp_logits = torch.exp(dot_contrast) * logits_mask.detach()
        log_prob = dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-16)

        mean_log_prob_pos = (label.detach() * log_prob).sum(1) / (label.sum(1) + 1e-16)

        if min(mean_log_prob_pos.shape) == 0:
            mean_log_prob_pos = mean_log_prob_pos * 0

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean() * self.feat_loss_weight

        return loss

    def mask_contrastive(self, mask_anchors, mask_keys, label):
        '''
        mask_anchors:   [n, c]
        mask_keys:      [n, k, c]
        label:          [n]
        '''
        mask_keys = mask_keys.flatten(0, 1)                         # [n * k, c]
        repeat_num = mask_keys.shape[0] // mask_anchors.shape[0]
        mask_keys = torch.cat([mask_anchors, mask_keys], dim=0)     # [n * (k + 1), c]
        
        anchor_label = label.unsqueeze(1)
        anchor_label = torch.eq(anchor_label, anchor_label.T)       # [n, n]
        key_label = anchor_label.repeat(1, repeat_num)              # [n * k, n]
        
        logits_mask = ~torch.eye(label.shape[0], dtype=torch.bool, device=label.device)
        anchor_label = anchor_label & logits_mask
        tot_label = torch.cat([anchor_label, key_label], dim=1)     # [n, n * (k + 1)]
        neg_label = ~tot_label
        
        dot_contrast = torch.div(torch.einsum('nc,mc->nm', 
            mask_anchors.detach(), mask_keys), self.temperature)    # [n, n * (k + 1)]
        
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        dot_contrast = dot_contrast - logits_max.detach()
        
        exp_logits = torch.exp(dot_contrast)
        log_prob = dot_contrast - torch.log((exp_logits * neg_label.detach()).sum(1, keepdim=True) + 1e-16)
        mean_log_prob = (tot_label.detach() * log_prob).sum(1) / (tot_label.sum(1) + 1e-16)
        
        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        loss = loss.mean() * self.mask_loss_weight
        return loss
    
    def forward(self,
                proj_feats,
                seg_logits,
                groundtruth,
                proj_masks):
        B, C, H, W = proj_feats.shape
        groundtruth = F.interpolate(
            input=groundtruth.float(),
            size=[H, W],
            mode='nearest').long()
        proj_masks = [F.normalize(x, dim=1) for x in proj_masks]
        proj_masks = [F.interpolate(
            input=mask_pred, size = [H, W],
            mode='bilinear', align_corners=True
        ) for mask_pred in proj_masks]
        
        proj_masks = [torch.einsum('bcn->bnc', mask_pred.flatten(2)) 
                          for mask_pred in proj_masks]                      # [b, h * w, c] * k
        
        groundtruth = groundtruth.flatten(1)                                # [b, h * w]
        seg_logits = seg_logits.softmax(dim=1).flatten(2).transpose(-2, -1) # [b, h * w, cls_num]
        predict = torch.argmax(seg_logits, dim=2)                           # [b, h * w]
        proj_feats = F.normalize(proj_feats.flatten(2), 
                        dim=1).transpose(-2, -1)                            # [b, h * w, c]

        features, labels, mask_anchors, mask_keys = \
            self._hard_sample_mining(predict, groundtruth, proj_feats, seg_logits, proj_masks)
        
        if features is None:
            return {'loss_feat_contrast':proj_feats.mean() * 0, 
                    'loss_mask_contrast':torch.cat(proj_masks, dim=0).mean() * 0}
        
        loss_feat_contrast = self.feat_contrastive(features, labels)
        loss_mask_contrast = self.mask_contrastive(mask_anchors, mask_keys, labels)
        
        return {'loss_feat_contrast':loss_feat_contrast, 'loss_mask_contrast':loss_mask_contrast}

    @property
    def loss_name(self):
        return self._loss_name

if __name__ == '__main__':
    Loss = ContrastLoss()
    predict = torch.rand((1, 19, 128, 128))
    label = torch.randint(0, 19, (1, 1, 512, 512))
    feature_map = torch.rand((1, 256, 128, 128))
    proj_masks = [torch.rand((1, 256, 128, 128)) for x in range(5)]
    loss = Loss(feature_map, predict, label, proj_masks)
    print(loss)
