import torch
import numpy as np

from pcdet.models.model_utils import model_nms_utils

try:
    import kornia
except:
    pass


def filter_boxes(batch_dict, cfgs):
    batch_size = batch_dict['batch_size']
    pred_dicts = []
    for index in range(batch_size):
        if batch_dict.get('batch_index', None) is not None:
            assert batch_dict['batch_box_preds'].shape.__len__() == 2
            batch_mask = (batch_dict['batch_index'] == index)
        else:
            assert batch_dict['batch_box_preds'].shape.__len__() == 3
            batch_mask = index

        box_preds = batch_dict['batch_box_preds'][batch_mask]
        cls_preds = batch_dict['batch_cls_preds'][batch_mask]

        if not batch_dict['cls_preds_normalized']:
            cls_preds = torch.sigmoid(cls_preds)

        max_cls_preds, label_preds = torch.max(cls_preds, dim=-1)
        if batch_dict.get('has_class_labels', False):
            label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
            label_preds = batch_dict[label_key][index]
        else:
            label_preds = label_preds + 1

        final_boxes = box_preds
        final_labels = label_preds
        final_cls_preds = cls_preds

        if cfgs.get('FILTER_BY_NMS', False):
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=max_cls_preds, box_preds=final_boxes,
                nms_config=cfgs.NMS.NMS_CONFIG,
                score_thresh=cfgs.NMS.SCORE_THRESH
            )

            final_labels = final_labels[selected]
            final_boxes = final_boxes[selected]
            final_cls_preds = final_cls_preds[selected]
            max_cls_preds = max_cls_preds[selected]

        if cfgs.get('FILTER_BY_SCORE_THRESHOLD', False):
            selected = max_cls_preds > cfgs.SCORE_THRESHOLD
            final_labels = final_labels[selected]
            final_boxes = final_boxes[selected]
            final_cls_preds = final_cls_preds[selected]
            max_cls_preds = max_cls_preds[selected]

        if cfgs.get('FILTER_BY_TOPK', False):
            topk = min(max_cls_preds.shape[0], cfgs.TOPK)
            selected = torch.topk(max_cls_preds, topk)[1]
            final_labels = final_labels[selected]
            final_boxes = final_boxes[selected]
            final_cls_preds = final_cls_preds[selected]
            max_cls_preds = max_cls_preds[selected]

        # added filtering boxes with size 0
        zero_mask = (final_boxes[:, 3:6] != 0).all(1)
        final_boxes = final_boxes[zero_mask]
        final_labels = final_labels[zero_mask]
        final_cls_preds = final_cls_preds[zero_mask]

        record_dict = {
            'pred_boxes': final_boxes,
            'pred_cls_preds': final_cls_preds,
            'pred_labels': final_labels
        }
        pred_dicts.append(record_dict)

    return pred_dicts