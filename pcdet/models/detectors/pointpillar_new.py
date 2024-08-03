from .detector3d_template import Detector3DTemplate
from pcdet.config import cfg
import torch

class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, batch_dict_teacher=None, return_batch_dict=False):
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        # for cur_module in self.module_list:
        #     if self.training or cur_module.__class__.__name__ not in ['PillarHeadMT',]:
        #         batch_dict = cur_module(batch_dict)
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        
        if return_batch_dict:
            return batch_dict

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            
            if batch_dict_teacher is not None:
                # loss1, tb_dict1, disp_dict1 = self.get_training_loss()
                loss2 = 0.0
                if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('CONSISTENCY', None) and cfg.SELF_TRAIN.CONSISTENCY.OBJECT_LOSS_WEIGHT > 0:
                    loss2, tb_dict2 = self.get_object_loss(batch_dict, batch_dict_teacher)
                    tb_dict.update(tb_dict2)
                tb_dict['loss'] = loss.item() if isinstance(loss, torch.Tensor) else loss

                loss = loss * cfg.SELF_TRAIN.CONSISTENCY.CONSISTENCY_LOSS_WEIGHT + loss2 * cfg.SELF_TRAIN.CONSISTENCY.OBJECT_LOSS_WEIGHT

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict, batch_dict
        else:
            # pred_dicts, recall_dicts = self.post_processing(batch_dict)
            pred_dicts, recall_dicts = self.post_processing_multicriterion(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, weights=None):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss(weights=None)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict