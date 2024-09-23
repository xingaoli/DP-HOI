# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from .dn_components import prepare_for_dn, dn_post_process, compute_dn_loss
from .clip_model import build_clip_caption_loss


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        # dn
        self.label_enc_od = nn.Embedding(num_classes + 1, hidden_dim - 1)
        self.bbox_enc_od = MLP(4, hidden_dim, hidden_dim, 2)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        # action
        self.verb_class_kinetics700 = nn.Linear(hidden_dim, 700)
        self.verb_class_haa500 = nn.Linear(hidden_dim, 500)
        self.maxpooling = nn.AdaptiveMaxPool1d(1)
        self.maxpooling2 = nn.MaxPool1d(16)
        # caption
        self.action_class_fc = nn.Sequential(
                nn.Linear(256, 512),
                nn.LayerNorm(512),) 

    def forward(self, samples: NestedTensor, samples_k700 = None, samples_haa500 = None, samples_caption = None, dn_args = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        b_d = src.size(0)
        input_query_label, input_query_bbox, attn_mask, mask_dict = \
            prepare_for_dn(dn_args, self.query_embed.weight, b_d, self.training, self.num_queries, self.num_classes,
                        self.hidden_dim, self.label_enc_od, self.bbox_enc_od)
        dataset = "detection"
        hs = self.transformer(self.input_proj(src), mask, dataset, input_query_bbox, pos[-1],    
                                    tgt=input_query_label, attn_mask=attn_mask)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_class, outputs_coord = dn_post_process(outputs_class, outputs_coord, mask_dict)
        if not self.training:
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            return out
        if samples_k700 is not None:
            dataset = "action"
            if isinstance(samples_k700, (list, torch.Tensor)):
                samples_k700 = nested_tensor_from_tensor_list(samples_k700)
            features, pos = self.backbone(samples_k700)
            src, mask = features[-1].decompose()
            hs_verb = self.transformer(self.input_proj(src), mask, dataset, self.query_embed.weight, pos[-1],
                                                class_embed=self.class_embed)            
            outputs_action_class = self.verb_class_kinetics700(hs_verb)
            #  max pooling
            outputs_action_class = torch.stack([self.maxpooling(i.permute(0, 2, 1)) for i in outputs_action_class]).permute(0, 1, 3, 2)  
            outputs_action_class = torch.stack([self.maxpooling2(i.permute(2, 1, 0)) for i in outputs_action_class]).permute(0, 3, 2, 1)          
        elif samples_haa500 is not None:
            dataset = "action"
            if isinstance(samples_haa500, (list, torch.Tensor)):
                samples_haa500 = nested_tensor_from_tensor_list(samples_haa500)
            features, pos = self.backbone(samples_haa500)
            src, mask = features[-1].decompose()
            hs_verb = self.transformer(self.input_proj(src), mask, dataset, self.query_embed.weight, pos[-1],
                                                class_embed=self.class_embed)            
            outputs_action_class = self.verb_class_haa500(hs_verb)
            outputs_action_class = torch.stack([self.maxpooling(i.permute(0, 2, 1)) for i in outputs_action_class]).permute(0, 1, 3, 2)  
        if samples_caption is not None:
            dataset = 'caption'
            if isinstance(samples_caption, (list, torch.Tensor)):
                samples_caption = nested_tensor_from_tensor_list(samples_caption)
            features, pos = self.backbone(samples_caption)
            src, mask = features[-1].decompose()
            hs_verb = self.transformer(self.input_proj(src), mask, dataset, self.query_embed.weight, pos[-1],
                                                class_embed=self.class_embed)            
            outputs_caption = []
            for hs_verb0 in hs_verb:
                outputs_caption.append(self.action_class_fc(hs_verb0))
            outputs_caption_embed = []
            for j in range(hs.shape[0]):
                outputs_caption_embed.append([i[j] for i in outputs_caption])
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_action_logits': outputs_action_class[-1], 'pred_caption_logits': outputs_caption_embed[-1]}
        out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_action_class, outputs_caption_embed)
        return out, mask_dict
        
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_action_class=None, outputs_caption_embed=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_action_class is None:
            return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        else:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_action_logits': c, 'pred_caption_logits': d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_action_class[:-1], outputs_caption_embed[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, clip_caption_loss):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.clip_caption_loss = clip_caption_loss

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    '''focal loss, more general version, can supervise more q'''
    def loss_action_labels(self, outputs, targets, indices=None, num_interactions=None):
        assert 'pred_action_logits' in outputs
        src_logits = outputs['pred_action_logits']
        bs = src_logits.shape[0]
        num_q = src_logits.shape[1]
        src_logits = src_logits.contiguous().view(bs*num_q, -1)
        idx = []
        for i, tgt in enumerate(targets):
            if len(tgt['action_labels']) != 0:
                idx.extend([u + i * num_q for u in range(num_q)])
        target_classes = []
        for t in targets:
            target_classes.append(t['action_labels'].repeat(num_q, 1))
        src_logits = src_logits.sigmoid()
        loss_action_ce = self._neg_loss(src_logits, target_classes, (targets, idx, num_q))
        losses = {'loss_action_bce': loss_action_ce}
        return losses

    def _neg_loss(self, pred, gt, targets_action_args, weights=None, alpha=0.25):
        loss = 0
        targets_action = targets_action_args[0]
        idx = targets_action_args[1]
        num_q = targets_action_args[2]
        num_pos = 0
        pos_loss = 0
        neg_loss = 0
        for i, t in enumerate(targets_action):
            gt_need = gt[i]
            pred_need = pred[idx[i*num_q:(i+1)*num_q],:]
            pos_inds = gt_need.eq(1).float()
            neg_inds = gt_need.lt(1).float()
            pos_loss_tmp = alpha * torch.log(pred_need) * torch.pow(1 - pred_need, 2) * pos_inds
            if weights is not None:
                pos_loss = pos_loss * weights[:-1]

            neg_loss_tmp = (1 - alpha) * torch.log(1 - pred_need) * torch.pow(pred_need, 2) * neg_inds
            num_pos += pos_inds.float().sum()
            pos_loss += pos_loss_tmp.sum()
            neg_loss += neg_loss_tmp.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def loss_caption_labels(self, outputs, targets, indices=None, num_boxes=None):
        assert 'pred_caption_logits' in outputs
        src_logits = outputs['pred_caption_logits']
        caption_gts = {}
        caption_gts['caption'] = [x['triplet_captions'] for x in targets]
        caption_gts['cluster_category'] = [x['cluster_category'] for x in targets]
        caption_gts['dataset_name'] = [x['dataset_name'] for x in targets]        
        loss_caption_infonce, caption_class_error = self.clip_caption_loss(src_logits, caption_gts)
        losses = {'loss_caption_ce': loss_caption_infonce, 'caption_class_error': caption_class_error}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices=None, num_boxes=None, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'action_labels': self.loss_action_labels,
            'caption_labels': self.loss_caption_labels, 
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, targets_action=None, targets_caption=None, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        if not self.training:
            cur_losses = ['labels', 'boxes', 'cardinality']
            for loss in cur_losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        else:
            cur_losses = ['labels', 'boxes', 'cardinality','action_labels','caption_labels']
            for loss in cur_losses:
                if loss == 'action_labels':
                    losses.update(self.get_loss(loss, outputs, targets_action)) 
                elif loss == 'caption_labels':
                    losses.update(self.get_loss(loss, outputs, targets_caption, indices, num_boxes))         
                else:
                    losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in cur_losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict) 
                    if not self.training:
                        continue
                    if loss == 'action_labels':
                        l_dict = self.get_loss(loss, aux_outputs, targets_action, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)  
                    elif loss == 'caption_labels':  
                        l_dict = self.get_loss(loss, aux_outputs, targets_caption, indices, num_boxes)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)  

        # dn loss computation
        aux_num = 0
        if 'aux_outputs' in outputs:
            aux_num = len(outputs['aux_outputs'])
        dn_losses = compute_dn_loss(mask_dict, self.training, aux_num)
        losses.update(dn_losses)
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    clip_caption_loss = build_clip_caption_loss(args)
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_action_bce'] = args.action_loss_coef
    weight_dict['loss_caption_ce'] = args.caption_loss_coef
    # todo add gt loss coef
    weight_dict['loss_gt_ce'] = 1
    weight_dict['loss_gt_bbox'] = args.bbox_loss_coef
    weight_dict['loss_gt_giou'] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses,
                             clip_caption_loss=clip_caption_loss)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
