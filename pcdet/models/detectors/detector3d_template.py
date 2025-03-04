import os

import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils
import numpy as np
import torch.nn.functional as F
from ...utils import common_utils
from pcdet.config import cfg

class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels \
            if hasattr(backbone_3d_module, 'backbone_channels') else None
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            backbone_channels=model_info_dict['backbone_channels'] \
                                if 'backbone_channels' in model_info_dict else None,
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        spconv_matched_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            spconv_matched_state[key] = val
            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
        if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
            self.load_state_dict(spconv_matched_state)
        elif strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        
        # if strict:
        #     self.load_state_dict(update_model_state)
        # else:
        #     state_dict.update(update_model_state)
        #     self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch


    def post_processing_multicriterion(self, batch_dict, no_nms=False):
        """
        For 
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            iou_preds = batch_dict['batch_cls_preds'][batch_mask]
            cls_preds = batch_dict['roi_scores'][batch_mask]

            if isinstance(cls_preds, list):
                cls_preds = torch.cat(cls_preds).squeeze()
            else:
                cls_preds = cls_preds.squeeze()

            src_iou_preds = iou_preds
            src_box_preds = box_preds
            src_cls_preds = cls_preds
            assert iou_preds.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                iou_preds = torch.sigmoid(iou_preds)
                cls_preds = torch.sigmoid(cls_preds)

            # TODO
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                iou_preds, label_preds = torch.max(iou_preds, dim=-1)
                label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels',
                                                                                False) else label_preds + 1
                if isinstance(label_preds, list):
                    label_preds = torch.cat(label_preds, dim=0)

                if post_process_cfg.NMS_CONFIG.get('SCORE_WEIGHTS', None):
                    weight_iou = post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.iou
                    weight_cls = post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.cls

                if post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) == 'iou' or \
                        post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) is None:
                    nms_scores = iou_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'cls':
                    nms_scores = cls_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'hybrid_iou_cls':
                    assert weight_iou + weight_cls == 1
                    nms_scores = weight_iou * iou_preds + \
                                 weight_cls * cls_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'Harmonious_iou_cls':
                    assert weight_iou + weight_cls == 1
                    # iou_preds^weight_iou * cls_preds^weight_cls
                    nms_scores = torch.pow(iou_preds, weight_iou) * torch.pow(cls_preds, weight_cls)
                else:
                    raise NotImplementedError

                if no_nms:
                    selected = nms_scores > (post_process_cfg.SCORE_THRESH)
                    selected = torch.arange(len(nms_scores), device=nms_scores.device)[selected]
                    selected_scores = nms_scores[selected]
                else:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=nms_scores, box_preds=box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    raise NotImplementedError

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
                'pred_cls_scores': cls_preds[selected],
                'pred_iou_scores': iou_preds[selected],
            }

            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict


    def get_object_loss(self, batch_target1, batch_target2):
        '''
        batch_target1:
            'rois': roi,
            'roi_scores': roi_scores,
            'roi_labels': roi_labels,
            'roi_head_features': roi_head_features
        '''
        batch_target1['rois_mt'][:,:,:7], gt_assignment_list = common_utils.reverse_augmentation(batch_target1['rois_mt'][:,:,:7], batch_target1)
        batch_size = batch_target1['rois_mt'].shape[0]
        object_loss = 0
        for index in range(batch_size): 
            pred_box_a = batch_target1['rois_mt'][index,:,:7]     #[x, y, z, dx, dy, dz, heading]
            cls_scores_a = batch_target1['roi_scores_mt'][index] 
            iou_scores_a = batch_target1['roi_iou_scores_mt'][index]
            roi_head_features_a = batch_target1['roi_head_features_mt'][index,:,:]
            pred_box_b = batch_target2['rois_mt'][index,:,:7]
            cls_scores_b = batch_target2['roi_scores_mt'][index] 
            iou_scores_b = batch_target2['roi_iou_scores_mt'][index]
            roi_head_features_b = batch_target2['roi_head_features_mt'][index,:,:]

            if pred_box_a.shape[0] == 0 or pred_box_b.shape[0] == 0:
                continue

            iou_scores_a = torch.sigmoid(iou_scores_a)
            iou_scores_b = torch.sigmoid(iou_scores_b)
            cls_scores_a = torch.sigmoid(cls_scores_a)
            cls_scores_b = torch.sigmoid(cls_scores_b)
            iou_scores_a = iou_scores_a.squeeze()
            iou_scores_b = iou_scores_b.squeeze()
            cls_scores_a = cls_scores_a.squeeze()
            cls_scores_b = cls_scores_b.squeeze()
            if cfg.SELF_TRAIN.EMA_LEARNING.get('SCORE_WEIGHTS', None):
                weight_iou = cfg.SELF_TRAIN.EMA_LEARNING.SCORE_WEIGHTS.iou
                weight_cls = cfg.SELF_TRAIN.EMA_LEARNING.SCORE_WEIGHTS.cls
            if cfg.SELF_TRAIN.EMA_LEARNING.get('SCORE_TYPE', None) == 'iou' or \
                    cfg.SELF_TRAIN.EMA_LEARNING.get('SCORE_TYPE', None) is None:
                cls_scores_a = iou_scores_a
                cls_scores_b = iou_scores_b
            elif cfg.SELF_TRAIN.EMA_LEARNING.SCORE_TYPE == 'cls':
                cls_scores_a = cls_scores_a
                cls_scores_b = cls_scores_b
            elif cfg.SELF_TRAIN.EMA_LEARNING.SCORE_TYPE == 'hybrid_iou_cls':
                assert weight_iou + weight_cls == 1
                cls_scores_a = weight_iou * iou_scores_a + weight_cls * cls_scores_a
                cls_scores_b = weight_iou * iou_scores_b + weight_cls * cls_scores_b
            else:
                raise NotImplementedError



            if cfg.SELF_TRAIN.EMA_LEARNING.get('FILTER_BY_NMS', False):
                selected_a, selected_scores_a = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_scores_a, box_preds=pred_box_a,
                    nms_config=cfg.SELF_TRAIN.EMA_LEARNING.NMS_CONFIG,
                    score_thresh=cfg.SELF_TRAIN.EMA_LEARNING.NMS_CONFIG.SCORE_THRESH
                )
                # selected_a = selected_a.cpu().numpy()
                
                pred_box_a = pred_box_a[selected_a]
                cls_scores_a = cls_scores_a[selected_a]
                iou_scores_a = iou_scores_a[selected_a]
                roi_head_features_a = roi_head_features_a[selected_a]

                selected_b, selected_scores_b = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_scores_b, box_preds=pred_box_b,
                    nms_config=cfg.SELF_TRAIN.EMA_LEARNING.NMS_CONFIG,
                    score_thresh=cfg.SELF_TRAIN.EMA_LEARNING.NMS_CONFIG.SCORE_THRESH
                )
                # selected_b = selected_b.cpu().numpy()
                pred_box_b = pred_box_b[selected_b]
                cls_scores_b = cls_scores_b[selected_b]
                iou_scores_b = iou_scores_b[selected_b]
                roi_head_features_b = roi_head_features_b[selected_b]
            if pred_box_a.shape[0] == 0 or pred_box_b.shape[0] == 0:
                continue
            if cfg.SELF_TRAIN.EMA_LEARNING.get('FILTER_BY_SCORE_THRESHOLD', False):
                selected_a = cls_scores_a > cfg.SELF_TRAIN.EMA_LEARNING.SCORE_THRESHOLD
                selected_b = cls_scores_b > cfg.SELF_TRAIN.EMA_LEARNING.SCORE_THRESHOLD
                pred_box_a = pred_box_a[selected_a]
                cls_scores_a = cls_scores_a[selected_a]
                iou_scores_a = iou_scores_a[selected_a]
                roi_head_features_a = roi_head_features_a[selected_a]
                pred_box_b = pred_box_b[selected_b]
                cls_scores_b = cls_scores_b[selected_b]
                iou_scores_b = iou_scores_b[selected_b]
                roi_head_features_b = roi_head_features_b[selected_b]
            if pred_box_a.shape[0] == 0 or pred_box_b.shape[0] == 0:
                continue
            if cfg.SELF_TRAIN.EMA_LEARNING.get('FILTER_BY_TOPK', False):
                topk = min(cfg.SELF_TRAIN.EMA_LEARNING.TOPK, pred_box_a.shape[0])
                selected_a = torch.topk(cls_scores_a, topk)[1]
                pred_box_a = pred_box_a[selected_a]
                cls_scores_a = cls_scores_a[selected_a]
                iou_scores_a = iou_scores_a[selected_a]
                roi_head_features_a = roi_head_features_a[selected_a]
                topk = min(cfg.SELF_TRAIN.EMA_LEARNING.TOPK, pred_box_b.shape[0])
                selected_b = torch.topk(cls_scores_b, topk)[1]
                pred_box_b = pred_box_b[selected_b]
                cls_scores_b = cls_scores_b[selected_b]
                iou_scores_b = iou_scores_b[selected_b]
                roi_head_features_b = roi_head_features_b[selected_b]
            if pred_box_a.shape[0] == 0 or pred_box_b.shape[0] == 0:
                continue
            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_box_a[:, :7], pred_box_b[:, :7]).cpu()
            ious, match_idx = torch.max(iou_matrix, dim=1) #返回和当前box的iou最大的box
            ious, match_idx = ious.numpy(), match_idx.numpy()
            pred_box_a, pred_box_b = pred_box_a.cpu().numpy(), pred_box_b.cpu().numpy()

            match_pairs_idx = np.concatenate((
                np.array(list(range(pred_box_a.shape[0]))).reshape(-1, 1),
                match_idx.reshape(-1, 1)), axis=1) #(0,i_0),(1,i_1),(2,i_2)...

            #########################################################
            # filter matched pair boxes by IoU
            # if matching succeeded, use boxes with higher confidence
            #########################################################
            try:
                iou_mask = (ious >= cfg.SELF_TRAIN.CONSISTENCY.get('IOU_THRESHOLD', 0.1)) #筛选s&t的ground truth box的iou足够大的region
            except:
                iou_mask = (ious >= cfg.SELF_TRAIN.CONSISTENCY_SRC.get('IOU_THRESHOLD', 0.1))
            matching_selected = match_pairs_idx[iou_mask]  
            roi_head_features_a = roi_head_features_a[matching_selected[:, 0]] # 2.变量维度batch_size * n * feature_size?
            roi_head_features_b = roi_head_features_b[matching_selected[:, 1]]
            pred_box_a = pred_box_a[matching_selected[:, 0]]
            pred_box_b = pred_box_b[matching_selected[:, 1]]
            cls_scores_a = cls_scores_a[matching_selected[:, 0]]
            cls_scores_b = cls_scores_b[matching_selected[:, 1]]

            if roi_head_features_a.shape[0] == 0:
                continue
            roi_head_features_a = F.normalize(roi_head_features_a, dim=1) # normalize feature
            roi_head_features_b = F.normalize(roi_head_features_b, dim=1)

            object_loss = torch.exp(-torch.einsum('nc,nc->n', [roi_head_features_a, roi_head_features_b])).mean()



        object_loss /= batch_size
        
        # object_loss = cfg.SELF_TRAIN.CONSISTENCY.get('OBJECT_LOSS_WEIGHT', 1.0) * object_loss
        tb_dict={} 

        tb_dict.update({'object_loss': object_loss.item() if isinstance(object_loss, torch.Tensor) else object_loss})

        return object_loss, tb_dict


    def get_contrasive_loss(self, batch_target1, batch_target2):
        '''
        batch_target1:
            'rois': roi,
            'roi_scores': roi_scores,
            'roi_labels': roi_labels,
            'roi_head_features': roi_head_features
        '''
        batch_target1['rois_mt'][:,:,:7], gt_assignment_list = common_utils.reverse_augmentation(batch_target1['rois_mt'][:,:,:7], batch_target1)
        batch_size = batch_target1['rois_mt'].shape[0]
        object_loss = 0
        for index in range(batch_size): 
            pred_box_a = batch_target1['rois_mt'][index,:,:7]     #[x, y, z, dx, dy, dz, heading]
            cls_scores_a = batch_target1['roi_scores_mt'][index] 
            iou_scores_a = batch_target1['roi_iou_scores_mt'][index]
            roi_head_features_a = batch_target1['roi_head_features_mt'][index,:,:]
            pred_box_b = batch_target2['rois_mt'][index,:,:7]
            # print(pred_box_b.shape)
            cls_scores_b = batch_target2['roi_scores_mt'][index] 
            iou_scores_b = batch_target2['roi_iou_scores_mt'][index]
            roi_head_features_b = batch_target2['roi_head_features_mt'][index,:,:]

            if pred_box_a.shape[0] == 0 or pred_box_b.shape[0] == 0:
                continue

            iou_scores_a = torch.sigmoid(iou_scores_a)
            iou_scores_b = torch.sigmoid(iou_scores_b)
            cls_scores_a = torch.sigmoid(cls_scores_a)
            cls_scores_b = torch.sigmoid(cls_scores_b)
            iou_scores_a = iou_scores_a.squeeze()
            iou_scores_b = iou_scores_b.squeeze()
            cls_scores_a = cls_scores_a.squeeze()
            cls_scores_b = cls_scores_b.squeeze()
            if cfg.SELF_TRAIN.EMA_LEARNING.get('SCORE_WEIGHTS', None):
                weight_iou = cfg.SELF_TRAIN.EMA_LEARNING.SCORE_WEIGHTS.iou
                weight_cls = cfg.SELF_TRAIN.EMA_LEARNING.SCORE_WEIGHTS.cls
            if cfg.SELF_TRAIN.EMA_LEARNING.get('SCORE_TYPE', None) == 'iou' or \
                    cfg.SELF_TRAIN.EMA_LEARNING.get('SCORE_TYPE', None) is None:
                cls_scores_a = iou_scores_a
                cls_scores_b = iou_scores_b
            elif cfg.SELF_TRAIN.EMA_LEARNING.SCORE_TYPE == 'cls':
                cls_scores_a = cls_scores_a
                cls_scores_b = cls_scores_b
            elif cfg.SELF_TRAIN.EMA_LEARNING.SCORE_TYPE == 'hybrid_iou_cls':
                assert weight_iou + weight_cls == 1
                cls_scores_a = weight_iou * iou_scores_a + weight_cls * cls_scores_a
                cls_scores_b = weight_iou * iou_scores_b + weight_cls * cls_scores_b
            else:
                raise NotImplementedError
            
            # 获取 teacher 的背景特征
            selected_b_low = cls_scores_b <= cfg.SELF_TRAIN.EMA_LEARNING.NEG_THRESHOLD
            pred_box_b_low = pred_box_b[selected_b_low]
            cls_scores_b_low = cls_scores_b[selected_b_low]
            iou_scores_b_low = iou_scores_b[selected_b_low]
            roi_head_features_b_low = roi_head_features_b[selected_b_low]

            # 获取 teacher 的前景特征
            selected_b = cls_scores_b > cfg.SELF_TRAIN.EMA_LEARNING.SCORE_THRESHOLD
            pred_box_b = pred_box_b[selected_b]
            cls_scores_b = cls_scores_b[selected_b]
            iou_scores_b = iou_scores_b[selected_b]
            roi_head_features_b = roi_head_features_b[selected_b]

            roi_head_features_b_fg = roi_head_features_b.clone()

            if pred_box_b_low.shape[0] == 0 or pred_box_b.shape[0] == 0:
                continue



            if cfg.SELF_TRAIN.EMA_LEARNING.get('FILTER_BY_NMS', False):
                selected_a, selected_scores_a = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_scores_a, box_preds=pred_box_a,
                    nms_config=cfg.SELF_TRAIN.EMA_LEARNING.NMS_CONFIG,
                    score_thresh=cfg.SELF_TRAIN.EMA_LEARNING.NMS_CONFIG.SCORE_THRESH
                )
                # selected_a = selected_a.cpu().numpy()
                
                pred_box_a = pred_box_a[selected_a]
                cls_scores_a = cls_scores_a[selected_a]
                iou_scores_a = iou_scores_a[selected_a]
                roi_head_features_a = roi_head_features_a[selected_a]
                # print(pred_box_b.shape)
                selected_b, selected_scores_b = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_scores_b, box_preds=pred_box_b,
                    nms_config=cfg.SELF_TRAIN.EMA_LEARNING.NMS_CONFIG,
                    score_thresh=cfg.SELF_TRAIN.EMA_LEARNING.NMS_CONFIG.SCORE_THRESH
                )
                # selected_b = selected_b.cpu().numpy()
                pred_box_b = pred_box_b[selected_b]
                cls_scores_b = cls_scores_b[selected_b]
                iou_scores_b = iou_scores_b[selected_b]
                roi_head_features_b = roi_head_features_b[selected_b]
            if pred_box_a.shape[0] == 0 or pred_box_b.shape[0] == 0:
                continue
            if cfg.SELF_TRAIN.EMA_LEARNING.get('FILTER_BY_SCORE_THRESHOLD', False):
                selected_a = cls_scores_a > cfg.SELF_TRAIN.EMA_LEARNING.SCORE_THRESHOLD
                selected_b = cls_scores_b > cfg.SELF_TRAIN.EMA_LEARNING.SCORE_THRESHOLD
                pred_box_a = pred_box_a[selected_a]
                cls_scores_a = cls_scores_a[selected_a]
                iou_scores_a = iou_scores_a[selected_a]
                roi_head_features_a = roi_head_features_a[selected_a]
                pred_box_b = pred_box_b[selected_b]
                cls_scores_b = cls_scores_b[selected_b]
                iou_scores_b = iou_scores_b[selected_b]
                roi_head_features_b = roi_head_features_b[selected_b]
            if pred_box_a.shape[0] == 0 or pred_box_b.shape[0] == 0:
                continue
            if cfg.SELF_TRAIN.EMA_LEARNING.get('FILTER_BY_TOPK', False):
                topk = min(cfg.SELF_TRAIN.EMA_LEARNING.TOPK, pred_box_a.shape[0])
                selected_a = torch.topk(cls_scores_a, topk)[1]
                pred_box_a = pred_box_a[selected_a]
                cls_scores_a = cls_scores_a[selected_a]
                iou_scores_a = iou_scores_a[selected_a]
                roi_head_features_a = roi_head_features_a[selected_a]
                topk = min(cfg.SELF_TRAIN.EMA_LEARNING.TOPK, pred_box_b.shape[0])
                selected_b = torch.topk(cls_scores_b, topk)[1]
                pred_box_b = pred_box_b[selected_b]
                cls_scores_b = cls_scores_b[selected_b]
                iou_scores_b = iou_scores_b[selected_b]
                roi_head_features_b = roi_head_features_b[selected_b]
            if pred_box_a.shape[0] == 0 or pred_box_b.shape[0] == 0:
                continue
            if cfg.SELF_TRAIN.CONSISTENCY.get('REWEIGHT', True):
            # if reweight:
                roi_head_features_a = torch.einsum('nc,n->nc', [roi_head_features_a, cls_scores_a])
                roi_head_features_b = torch.einsum('nc,n->nc', [roi_head_features_b, cls_scores_b])

            if roi_head_features_a.shape[0] == 0:
                continue
            roi_head_features_a = F.normalize(roi_head_features_a, dim=1) # normalize feature
            
            negative_samples = torch.cat((roi_head_features_b_fg, roi_head_features_b_low), dim=0)

            negative_samples = F.normalize(negative_samples, dim=1)

            if roi_head_features_b_low.shape[0] != 0:
                roi_head_features_b_low = F.normalize(roi_head_features_b_low, dim=1)

            if roi_head_features_b_fg.shape[0] != 0:
                roi_head_features_b_fg = F.normalize(roi_head_features_b_fg, dim=1)

            tau = 0.07

            similarity_pos = torch.mm(roi_head_features_a, roi_head_features_b_fg.t()) / tau

            similarity_neg = torch.mm(roi_head_features_a, negative_samples.t()) / tau

            logsumexp_neg = torch.logsumexp(similarity_neg, dim=1, keepdim=True)

            losses = -torch.logsumexp(similarity_pos - logsumexp_neg, dim=1)

            inter_cls_loss = losses.mean()

            object_loss = inter_cls_loss


        object_loss /= batch_size
        
        tb_dict={} 

        tb_dict.update({'object_loss': object_loss.item() if isinstance(object_loss, torch.Tensor) else object_loss})

        return object_loss, tb_dict


    ## for mean teacher
    def get_graph_loss(self, batch_target1, batch_target2, reweight=False): 
        '''
        batch_target1:
            'rois_mt': roi,
            'roi_scores_mt': roi_scores_mt,
            'roi_labels_mt': roi_labels,
            'roi_head_features': roi_head_features
        '''
        reweight = False
        batch_size = batch_target1['rois'].shape[0]
        loss = 0
        inter_graph_loss = 0
        object_loss = 0
        GLR_loss = 0

        # if cfg.SELF_TRAIN.CONSISTENCY.get('AUG', None):
        batch_target1['rois'][:,:,:7], gt_assignment_list = common_utils.reverse_augmentation(batch_target1['rois'][:,:,:7], batch_target1)
        for index in range(batch_size): 
            pred_box_a = batch_target1['rois_mt'][index,:,:7]     #[x, y, z, dx, dy, dz, heading]
            cls_scores_a = batch_target1['roi_scores_mt'][index] 
            roi_head_features_a = batch_target1['roi_head_features_mt'][index,:,:]
            pred_box_b = batch_target2['rois_mt'][index,:,:7]
            cls_scores_b = batch_target2['roi_scores_mt'][index] 
            roi_head_features_b = batch_target2['roi_head_features_mt'][index,:,:]

            if pred_box_a.shape[0] == 0 or pred_box_b.shape[0] == 0:
                continue

            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_box_a[:, :7], pred_box_b[:, :7]).cpu()
            ious, match_idx = torch.max(iou_matrix, dim=1) #返回和当前box的iou最大的box
            ious, match_idx = ious.numpy(), match_idx.numpy()
            pred_box_a, pred_box_b = pred_box_a.cpu().numpy(), pred_box_b.cpu().numpy()

            match_pairs_idx = np.concatenate((
                np.array(list(range(pred_box_a.shape[0]))).reshape(-1, 1),
                match_idx.reshape(-1, 1)), axis=1) #(0,i_0),(1,i_1),(2,i_2)...

            #########################################################
            # filter matched pair boxes by IoU
            # if matching succeeded, use boxes with higher confidence
            #########################################################
            
            iou_mask = (ious >= self.model_cfg.get('INTER_GRAPH_THRESHOLD', 0.1)) #筛选s&t的ground truth box的iou足够大的region

            matching_selected = match_pairs_idx[iou_mask]  
            roi_head_features_a = roi_head_features_a[matching_selected[:, 0]] # 2.变量维度batch_size * n * feature_size?
            roi_head_features_b = roi_head_features_b[matching_selected[:, 1]]
            pred_box_a = pred_box_a[matching_selected[:, 0]]
            pred_box_b = pred_box_b[matching_selected[:, 1]]
            cls_scores_a = cls_scores_a[matching_selected[:, 0]]
            cls_scores_b = cls_scores_b[matching_selected[:, 1]]

            if roi_head_features_a.shape[0] == 0:
                continue
            roi_head_features_a = F.normalize(roi_head_features_a, dim=1) # normalize feature
            roi_head_features_b = F.normalize(roi_head_features_b, dim=1)


            if reweight:
                roi_head_features_a = torch.einsum('nc,n->nc', [roi_head_features_a, cls_scores_a])
                roi_head_features_b = torch.einsum('nc,n->nc', [roi_head_features_b, cls_scores_b])

            pred_box_a = torch.Tensor(pred_box_a).cuda()
            pred_box_b = torch.Tensor(pred_box_b).cuda()
            adjacent_matrix_a = torch.exp(-((pred_box_a.unsqueeze(0) - pred_box_a.unsqueeze(1))**2 * torch.Tensor([[[1,1,1,5,5,5,20]]]).cuda()).sum(2) / self.model_cfg.get('TEMPERATURE', 13.0)**2) # N*N
            adjacent_matrix_b = torch.exp(-((pred_box_b.unsqueeze(0) - pred_box_b.unsqueeze(1))**2 * torch.Tensor([[[1,1,1,5,5,5,20]]]).cuda()).sum(2) / self.model_cfg.get('TEMPERATURE', 13.0)**2)
            
            inter_graph_loss += torch.norm(adjacent_matrix_a - adjacent_matrix_b).mean()

            GLR_a = torch.diag(torch.einsum('cm,mn,nb->cb', [roi_head_features_a.T, adjacent_matrix_a, roi_head_features_a])).sum()
            GLR_b = torch.diag(torch.einsum('cm,mn,nb->cb', [roi_head_features_b.T, adjacent_matrix_b, roi_head_features_b])).sum()
            N = roi_head_features_b.shape[0]
            GLR_loss += torch.abs(GLR_a - GLR_b) / N**2


            object_loss = torch.exp(-torch.einsum('nc,nc->n', [roi_head_features_a, roi_head_features_b])).mean()

        inter_graph_loss /= batch_size
        object_loss /= batch_size
        GLR_loss /= batch_size
        tb_dict={} 
        if inter_graph_loss == 0:
            return 0, {'inter_graph_loss': 0, 'object_loss': 0}
        if self.model_cfg.CONSISTENCY_LOSS['inter_graph_loss_weight'] > 0:
            gamma = 1 - self.model_cfg.CONSISTENCY_LOSS.get('GLR_loss_weight', 0)
            loss += (inter_graph_loss * gamma + GLR_loss * (1-gamma)) * self.model_cfg.CONSISTENCY_LOSS['inter_graph_loss_weight']
        tb_dict.update({'inter_graph_loss': inter_graph_loss.item()})   
        tb_dict.update({'GLR_loss': GLR_loss.item()})   

        if self.model_cfg.CONSISTENCY_LOSS.get('object_loss_weight', 0) > 0:
            loss += object_loss * self.model_cfg.CONSISTENCY_LOSS['object_loss_weight']
        tb_dict.update({'object_loss': object_loss.item()})

        return loss, tb_dict

    def split_batch_dicts(self, batch_dict):
        # split source and target dict for proposal generation and target assignments
        batch_source = {}
        batch_target = {}
        for key, val in batch_dict.items():
            if key == 'batch_size':
                batch_size = val // 2
                batch_source[key], batch_target[key] = batch_size, batch_size
            elif key in ['batch_type', 'cls_preds_normalized', 
                            'has_class_labels', 'dataset_cfg',
                            'encoded_spconv_tensor', 'encoded_spconv_tensor_stride',
                            'multi_scale_3d_features','multi_scale_3d_strides',
                            'spatial_features_stride', ]:
                batch_source[key], batch_target[key] = val, val
            # elif key == 'gt_boxes':
            #     batch_source[key] = val
            elif key in ['pos_ps_bbox', 'ign_ps_bbox', 'world_flip_along_x', 'world_flip_along_y', 'world_rotation', 'world_scaling', 'object_rotate_noise', 'object_scale_noise']:
                batch_target[key] = val
            elif key in ['points', ]:
                batch_size = batch_dict['batch_size'] // 2
                batch_source[key] = val[val[:,0] < batch_size,:]
                batch_target[key] = val[val[:,0] >= batch_size,:]
                batch_target['points'][:,0] -= batch_size
            elif key in ['voxels', 'voxel_coords', 'voxel_features', 'voxel_num_points']:
                continue
            else:
                # print('key', key, 'val', val.shape)
                split_length = val.shape[0] // 2
                batch_source[key], batch_target[key] = val[:split_length], val[split_length:]
    
        return batch_source, batch_target