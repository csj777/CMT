import torch
import torch.nn.functional as F
import os
import glob
import tqdm
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils
from pcdet.utils import self_training_utils
from pcdet.config import cfg
from pcdet.models.model_utils.dsnorm import set_ds_source, set_ds_target
from pcdet.models import load_data_to_gpu
from .train_utils import save_checkpoint, checkpoint_state
from pcdet.datasets.augmentor.augmentor_utils import get_points_in_box
from pcdet.utils.box_utils import remove_points_in_boxes3d, enlarge_box3d, \
    boxes3d_kitti_lidar_to_fakelidar, boxes_to_corners_3d
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import copy
import numpy as np
import wandb
import random

def train_one_epoch_st(model, optimizer, source_reader, target_loader, model_func, lr_scheduler,
                       accumulated_iter, optim_cfg, rank, tbar, total_it_each_epoch,
                       dataloader_iter, tb_log=None, leave_pbar=False, ema_model=None, cur_epoch=None):
    if total_it_each_epoch == len(target_loader):
        dataloader_iter = iter(target_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    ps_bbox_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    ign_ps_bbox_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    loss_meter = common_utils.AverageMeter()
    st_loss_meter = common_utils.AverageMeter()

    consistency_loss_meter = common_utils.AverageMeter()
    object_loss_meter = common_utils.AverageMeter()
    consistency_loss_meter_src = common_utils.AverageMeter()
    object_loss_meter_src = common_utils.AverageMeter()

    disp_dict = {}

    for cur_it in range(total_it_each_epoch):
        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        if ema_model is not None:
            ema_model.train()

        optimizer.zero_grad()

        source_batch = source_reader.read_data()
        try:
            target_batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(target_loader)
            target_batch = next(dataloader_iter)
            print('new iters')
        
        if cfg.SELF_TRAIN.get('MT', None):
            batch_target1, batch_target2 = target_batch # batch_target1: student, batch_target2: teacher

            # add tag for target domain
            batch_target1['batch_type'] = 'target'
            batch_target2['batch_type'] = 'target'
            
            ema_model.train()
            if cfg.SELF_TRAIN.get('DSNORM', None):
                ema_model.apply(set_ds_target)

            target_batch = batch_target1

        if cfg.SELF_TRAIN.SRC.USE_DATA:
            if cfg.SELF_TRAIN.get('DSNORM', None):
                model.apply(set_ds_source)
            model.train()
            loss, tb_dict, disp_dict, batch_dict_source = model_func(model, source_batch)
            loss = cfg.SELF_TRAIN.SRC.get('LOSS_WEIGHT', 1.0) * loss
            loss.backward()
            loss_meter.update(loss.item())
            disp_dict.update({'loss': "{:.3f}({:.3f})".format(loss_meter.val, loss_meter.avg)})

        if not cfg.SELF_TRAIN.SRC.get('USE_GRAD', None):
            optimizer.zero_grad()

        if cfg.SELF_TRAIN.TAR.USE_DATA:
            if cfg.SELF_TRAIN.get('ADABN', None) and cfg.get('MT', None) == None:
                target_batch_raw = copy.deepcopy(target_batch)
                load_data_to_gpu(target_batch_raw)
                if ema_model is not None:
                    ema_model.train()
                    if cfg.SELF_TRAIN.get('DSNORM', None):
                        ema_model.apply(set_ds_target)
                    with torch.no_grad():
                        batch_dict_teacher_ = ema_model(target_batch_raw, return_batch_dict=True)
                else:
                    model.train()
                    if cfg.SELF_TRAIN.get('DSNORM', None):
                        model.apply(set_ds_target)
                    with torch.no_grad():
                        batch_dict_teacher_ = model(target_batch_raw, return_batch_dict=True)

            if cfg.SELF_TRAIN.get('DSNORM', None):
                model.apply(set_ds_target)
            model.train()
            
            if cfg.SELF_TRAIN.get('OBJECT', None) and source_batch['gt_boxes'].shape[0] == target_batch['gt_boxes'].shape[0]:
                data_dict_list = []
                data_dict_list_tea = []
                batch_size = min(target_batch['gt_boxes'].shape[0], source_batch['gt_boxes'].shape[0])

                for idx in range(batch_size):
                    mix_data = {}
                    try:
                        s_gt_box = source_batch['gt_boxes'][idx].cpu().numpy()
                    except:
                        s_gt_box = source_batch['gt_boxes'][idx]
                    try:
                        t_gt_box = target_batch['gt_boxes'][idx].cpu().numpy()
                    except:
                        t_gt_box = target_batch['gt_boxes'][idx]

                    try:
                        t_gt_box_teacher = batch_target2['gt_boxes'][idx].cpu().numpy()
                    except:
                        t_gt_box_teacher = batch_target2['gt_boxes'][idx]

                    try:
                        single_pc_pnts = \
                            source_batch['points'][
                                source_batch['points'][:, 0] == idx][:, 1:].cpu().numpy()
                    except:
                        single_pc_pnts = \
                            source_batch['points'][
                                source_batch['points'][:, 0] == idx][:, 1:]

                    num_source_box_points = []
                    # combine s_gt_box and PC points
                    ps_pnts_to_sample_src = None
                    for box in s_gt_box:
                        points_in_box, mask = get_points_in_box(single_pc_pnts, box[:7])
                        ps_pnts_to_sample_src = points_in_box if ps_pnts_to_sample_src is None else np.concatenate([ps_pnts_to_sample_src, points_in_box])
                        num_source_box_points.append(points_in_box.shape[0])
                    num_source_box_points = np.array(num_source_box_points)

                    try:
                        target_points = target_batch['points'][target_batch['points'][:, 0] == idx][:, 1:].cpu().numpy()
                    except:
                        target_points = target_batch['points'][target_batch['points'][:, 0] == idx][:, 1:]

                    try:
                        target_points_teacher = batch_target2['points'][batch_target2['points'][:, 0] == idx][:, 1:].cpu().numpy()
                    except:
                        target_points_teacher = batch_target2['points'][batch_target2['points'][:, 0] == idx][:, 1:]
                    
                    num_target_box_points = []
                    # combine t_gt_box and PC points
                    ps_pnts_to_sample = None
                    for box in t_gt_box:
                        points_in_box, mask = get_points_in_box(target_points, box[:7])
                        ps_pnts_to_sample = points_in_box if ps_pnts_to_sample is None else np.concatenate([ps_pnts_to_sample, points_in_box])
                        num_target_box_points.append(points_in_box.shape[0])
                    num_target_box_points = np.array(num_target_box_points)


                    
                    if cfg.SELF_TRAIN.OBJECT.get('HARD', None) and cfg.SELF_TRAIN.OBJECT.HARD:
                        # mean_target_box_points = np.mean(num_target_box_points)
                        threshold = cfg.SELF_TRAIN.OBJECT.threshold
                        mean_target_box_points = np.percentile(num_target_box_points, threshold)

                        hard_source_box_idx = np.where(num_source_box_points < mean_target_box_points)[0]

                        if len(hard_source_box_idx) < 5:
                            hard_source_box_idx = np.argsort(num_source_box_points)[:1]
                        
                        max_num = cfg.SELF_TRAIN.OBJECT.get('max', 15)
                        if len(hard_source_box_idx) > max_num:
                            hard_source_box_idx = np.random.choice(hard_source_box_idx, max_num, replace=False)
                        
                        s_gt_box = [s_gt_box[i] for i in hard_source_box_idx]
                        s_gt_box = np.array(s_gt_box)

                        from pcdet.ops.iou3d_nms import iou3d_nms_utils


                        s_gt_box, _ = common_utils.check_numpy_to_torch(s_gt_box)
                        t_gt_box, _ = common_utils.check_numpy_to_torch(t_gt_box)
                        iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(s_gt_box[:, :7], t_gt_box[:, :7]).cpu().numpy()
                        s_gt_box = s_gt_box.cpu().numpy()
                        t_gt_box = t_gt_box.cpu().numpy()
                        related_boxes = iou_matrix > 0.01
                        # 选出 与 target gt_box 无关的 source gt_box
                        s_gt_box_stu = s_gt_box[np.sum(related_boxes, axis=1) == 0]
                        s_gt_box_stu = np.array(s_gt_box_stu)

                        s_gt_box, _ = common_utils.check_numpy_to_torch(s_gt_box)
                        t_gt_box_teacher, _ = common_utils.check_numpy_to_torch(t_gt_box_teacher)
                        iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(s_gt_box[:, :7], t_gt_box_teacher[:, :7]).cpu().numpy()
                        s_gt_box = s_gt_box.cpu().numpy()
                        t_gt_box_teacher = t_gt_box_teacher.cpu().numpy()
                        related_boxes = iou_matrix > 0.01
                        # 选出 与 target gt_box 无关的 source gt_box
                        s_gt_box_tea = s_gt_box[np.sum(related_boxes, axis=1) == 0]
                        s_gt_box_tea = np.array(s_gt_box_tea)


                    s_gt_box = s_gt_box_stu
                    ps_pnts_to_sample = None
                    for box in s_gt_box:
                        points_in_box, mask = get_points_in_box(single_pc_pnts, box[:7])
                        ps_pnts_to_sample = points_in_box if ps_pnts_to_sample is None else np.concatenate([ps_pnts_to_sample, points_in_box])
                    
                    target_points = remove_points_in_boxes3d(target_points, enlarge_box3d(s_gt_box[:, :7], extra_width=[1, 0.5, 0.5]))

                    try:
                        target_points = np.concatenate([target_points, ps_pnts_to_sample])
                    except:
                        pass

                    ps_pnts_to_sample = None
                    for box in s_gt_box_tea:
                        points_in_box, mask = get_points_in_box(single_pc_pnts, box[:7])
                        ps_pnts_to_sample = points_in_box if ps_pnts_to_sample is None else np.concatenate([ps_pnts_to_sample, points_in_box])

                    target_points_teacher = remove_points_in_boxes3d(target_points_teacher, enlarge_box3d(s_gt_box_tea[:, :7], extra_width=[1, 0.5, 0.5]))
                    try:
                        target_points_teacher = np.concatenate([target_points_teacher, ps_pnts_to_sample])
                    except:
                        pass

                    n_hard_box = len(s_gt_box)
                    t_gt_box = np.concatenate([t_gt_box, s_gt_box])
                    # 将 新的points和gt_boxes 和 target_batch 原有的key结合
                    data_dict = {'points': target_points, 'frame_id': target_batch['frame_id'][idx], 'gt_boxes': t_gt_box[:, :7], 'n_hard_box': n_hard_box}#, 'gt_names': gt_names}
                    
                    # prepare_data_easy: only transform_points_to_voxels
                    data_dict = target_loader.dataset.prepare_data_easy(data_dict=data_dict)
                    data_dict_list.append(data_dict)

                    t_gt_box_teacher = np.concatenate([t_gt_box_teacher, s_gt_box_tea])
                    data_dict = {'points': target_points_teacher, 'frame_id': target_batch['frame_id'][idx], 'gt_boxes': t_gt_box_teacher[:, :7], 'n_hard_box': n_hard_box}#, 'gt_names': gt_names}
                    data_dict = target_loader.dataset.prepare_data_easy(data_dict=data_dict)
                    data_dict_list_tea.append(data_dict)
            
                target_batch_new = target_loader.dataset.collate_batch(data_dict_list)
                for key in target_batch.keys():
                    if key not in target_batch_new:
                        target_batch_new[key] = target_batch[key]
                target_batch = target_batch_new

                target_batch_new_tea = target_loader.dataset.collate_batch(data_dict_list_tea)
                for key in batch_target2.keys():
                    if key not in target_batch_new_tea:
                        target_batch_new_tea[key] = batch_target2[key]
                batch_target2 = target_batch_new_tea

            if cfg.SELF_TRAIN.get('CONSISTENCY', None):
                if cfg.SELF_TRAIN.CONSISTENCY.get('SCENE', None):
                    if cfg.SELF_TRAIN.CONSISTENCY.get('SCENE_TRAIN', False):
                        target_batch_raw = copy.deepcopy(target_batch)
                        load_data_to_gpu(target_batch_raw)
                        if ema_model:
                            if cfg.SELF_TRAIN.DSNORM:
                                ema_model.apply(set_ds_target)
                            ema_model.train()
                            with torch.no_grad():
                                batch_dict_teacher = ema_model(target_batch_raw, return_batch_dict=True)
                                
                    if cfg.SELF_TRAIN.CONSISTENCY.get('SCENE_EVAL', None):
                        target_batch_raw = copy.deepcopy(target_batch)
                        load_data_to_gpu(target_batch_raw)
                        if ema_model:
                            if cfg.SELF_TRAIN.DSNORM:
                                ema_model.apply(set_ds_target)
                            ema_model.eval()
                            with torch.no_grad():
                                batch_dict_teacher = ema_model(target_batch_raw, return_batch_dict=True)
                    
                    # target_batch = batch_target1
                    
                    data_dict_list = []
                    batch_size = min(target_batch['gt_boxes'].shape[0], source_batch['gt_boxes'].shape[0])
                    for idx in range(batch_size):
                        target_idx = batch_size - idx - 1
                        try:
                            s_gt_box = source_batch['gt_boxes'][idx].cpu().numpy()
                        except:
                            s_gt_box = source_batch['gt_boxes'][idx]
                        try:
                            t_gt_box = target_batch['gt_boxes'][target_idx].cpu().numpy()
                        except:
                            t_gt_box = target_batch['gt_boxes'][target_idx]
                        try:
                            single_pc_pnts = \
                                source_batch['points'][
                                    source_batch['points'][:, 0] == idx][:, 1:].cpu().numpy()
                        except:
                            single_pc_pnts = \
                                source_batch['points'][
                                    source_batch['points'][:, 0] == idx][:, 1:]


                        # remove s_gt points_in_boxes in PC
                        single_pc_pnts = \
                            remove_points_in_boxes3d(single_pc_pnts, enlarge_box3d(s_gt_box[:, :7], extra_width=[1, 0.5, 0.5]))
                        
                        # remove t_gt points_in_boxes in PC
                        single_pc_pnts = \
                            remove_points_in_boxes3d(single_pc_pnts, enlarge_box3d(t_gt_box[:, :7], extra_width=[1, 0.5, 0.5]))
                        
                        try:
                            target_points = target_batch['points'][target_batch['points'][:, 0] == target_idx][:, 1:].cpu().numpy()
                        except:
                            target_points = target_batch['points'][target_batch['points'][:, 0] == target_idx][:, 1:]
                        # combine t_gt_box and PC points
                        ps_pnts_to_sample = None
                        for box in t_gt_box:
                            points_in_box, mask = get_points_in_box(target_points, box[:7])
                            ps_pnts_to_sample = points_in_box if ps_pnts_to_sample is None else np.concatenate([ps_pnts_to_sample, points_in_box])
                        # 结合 ps_pnts_to_sample 和 single_pc_pnts, 重新构建points
                        try:
                            single_pc_pnts = np.concatenate([ps_pnts_to_sample, single_pc_pnts])
                        except ValueError:
                            pass
                        
                        data_dict = {'points': single_pc_pnts, 'frame_id': target_batch['frame_id'][target_idx], 'gt_boxes': t_gt_box[:, :7]}#, 'gt_names': gt_names}
                        
                        data_dict = source_reader.dataloader.dataset.prepare_data_easy(data_dict=data_dict)
                        data_dict_list.append(data_dict)
                
                    data_dict = source_reader.dataloader.dataset.collate_batch(data_dict_list)
                    for key in target_batch.keys():
                        if key not in data_dict:
                            data_dict[key] = target_batch[key]
                    # batch_dict_teacher = {}
                    if batch_target2['gt_scores'] is not None:
                        batch_target2.pop('gt_scores')
                    # forward teacher model first
                    load_data_to_gpu(batch_target2)
                    if cfg.SELF_TRAIN.get('ADABN', None):
                        ema_model.train()
                        batch_target2 = ema_model(batch_target2, return_batch_dict=True)
                    else:
                        ema_model.eval()
                        batch_target2 = ema_model(batch_target2, return_batch_dict=True)
                    batch_dict_teacher = {}
                    for key in ['rois_mt', 'roi_head_features_mt', 'roi_scores_mt', 'roi_iou_scores_mt']:
                        batch_dict_teacher[key] = batch_target2[key].detach().clone()
                    
                    load_data_to_gpu(batch_dict_teacher)

                    if cfg.SELF_TRAIN.get('DSNORM', None):
                        model.apply(set_ds_source)
                        # model.apply(set_ds_target)
                    consistency_loss, consistency_tb_dict, consistency_disp_dict, consistency_batch_dict_target = model_func(model, data_dict, batch_dict_teacher)

                    consistency_loss.backward()

                    try:
                        object_loss_item = consistency_tb_dict['object_loss']
                    except:
                        object_loss_item = 0
                    try:
                        consistency_loss_item = consistency_tb_dict['loss']
                    except:
                        consistency_loss_item = 0
                    consistency_loss_meter.update(consistency_loss_item)
                    object_loss_meter.update(object_loss_item)
                    object_tb_dict = {}
                    object_tb_dict.update({'object_loss': object_loss_item})
                    consistency_tb_dict.update({'consistency_loss': consistency_loss_item})
                    consistency_disp_dict = {}
                    object_disp_dict = {}

            if cfg.SELF_TRAIN.get('DSNORM', None):
                if ema_model is not None:
                    ema_model.apply(set_ds_target)
                model.apply(set_ds_target)
            
            # parameters for save pseudo label on the fly
            st_loss, st_tb_dict, st_disp_dict, batch_dict_target = model_func(model, target_batch)
            st_loss = cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0) * st_loss
            st_loss.backward()

            st_loss_meter.update(st_loss.item())
            # count number of used ps bboxes in this batch
            pos_pseudo_bbox = target_batch['pos_ps_bbox'].mean(dim=0).cpu().numpy()
            ign_pseudo_bbox = target_batch['ign_ps_bbox'].mean(dim=0).cpu().numpy()
            ps_bbox_nmeter.update(pos_pseudo_bbox.tolist())
            ign_ps_bbox_nmeter.update(ign_pseudo_bbox.tolist())
            pos_ps_result = ps_bbox_nmeter.aggregate_result()
            ign_ps_result = ign_ps_bbox_nmeter.aggregate_result()

            st_tb_dict = common_utils.add_prefix_to_dict(st_tb_dict, 'st_')
            disp_dict.update(common_utils.add_prefix_to_dict(st_disp_dict, 'st_'))
            disp_dict.update({'st_loss': "{:.3f}({:.3f})".format(st_loss_meter.val, st_loss_meter.avg),
                              'pos_ps_box': pos_ps_result,
                              'ign_ps_box': ign_ps_result})
            if cfg.SELF_TRAIN.get('CONSISTENCY', None):
                consistency_tb_dict = common_utils.add_prefix_to_dict(consistency_tb_dict, 'consistency_')
                disp_dict.update(common_utils.add_prefix_to_dict(consistency_disp_dict, 'consistency_'))
                disp_dict.update({'consistency_loss': "{:.3f}({:.3f})".format(consistency_loss_meter.val, consistency_loss_meter.avg)})
                object_tb_dict = common_utils.add_prefix_to_dict(object_tb_dict, 'object_')
                disp_dict.update(common_utils.add_prefix_to_dict(object_disp_dict, 'object_'))
                disp_dict.update({'object_loss': "{:.3f}({:.3f})".format(object_loss_meter.val, object_loss_meter.avg)})
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        accumulated_iter += 1

        if ema_model is not None:
            update_ema_variables(model, ema_model, model_cfg=ema_model.model_cfg, cur_iter=accumulated_iter)
        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter, pos_ps_box=pos_ps_result,
                                  ign_ps_box=ign_ps_result))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                wandb.log({'meta_data/learning_rate': cur_lr},
                          step=int(accumulated_iter))
                if cfg.SELF_TRAIN.SRC.USE_DATA:
                    tb_log.add_scalar('train/loss', loss, accumulated_iter)
                    for key, val in tb_dict.items():
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)
                        wandb.log({key: val}, step=int(accumulated_iter))

                if cfg.SELF_TRAIN.TAR.USE_DATA:
                    tb_log.add_scalar('train/st_loss', st_loss, accumulated_iter)
                    for key, val in st_tb_dict.items():
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)
                        wandb.log({key: val}, step=int(accumulated_iter))

                if cfg.SELF_TRAIN.get('CONSISTENCY', None):
                    tb_log.add_scalar('train/consistency_loss', consistency_loss, accumulated_iter)
                    for key, val in consistency_tb_dict.items():
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)
                        wandb.log({key: val}, step=int(accumulated_iter))
                    for key, val in object_tb_dict.items():
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)
                        wandb.log({key: val}, step=int(accumulated_iter))


    if rank == 0:
        pbar.close()
        for i, class_names in enumerate(target_loader.dataset.class_names):
            tb_log.add_scalar(
                'ps_box/pos_%s' % class_names, ps_bbox_nmeter.meters[i].avg, cur_epoch)
            tb_log.add_scalar(
                'ps_box/ign_%s' % class_names, ign_ps_bbox_nmeter.meters[i].avg, cur_epoch)
            wandb.log({'ps_box/pos_%s' % class_names:
                           ps_bbox_nmeter.meters[i].avg})
            wandb.log({'ps_box/ign_%s' % class_names:
                           ign_ps_bbox_nmeter.meters[i].avg})    

    return accumulated_iter

def update_ema_variables(model, ema_model, model_cfg=None, cur_iter=0):
    assert model_cfg is not None

    multiplier = 1.0

    alpha = model_cfg['EMA_MODEL_ALPHA']
    alpha = 1 - multiplier*(1-alpha)
    if cfg.SELF_TRAIN.get('HSSDA_EMA', None):
        ema_keep_rate = 0.999
        change_global_step = 1000
        if cur_iter < change_global_step:
            keep_rate = (ema_keep_rate - 0.8) / change_global_step * cur_iter + 0.8
        else:
            keep_rate = ema_keep_rate
        alpha = keep_rate
        # Use the true average until the exponential average is more correct
        # alpha = min(1 - 1 / (cur_iter + 1), alpha)
        # for ema_param, param in zip(self.pv_rcnn_ema.parameters(), self.pv_rcnn.parameters()):
        #     ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    model_named_buffers = model.module.named_buffers() if hasattr(model, 'module') else model.named_buffers()
    for emabf, bf in zip(ema_model.named_buffers(), model_named_buffers):
        emaname, emavalue = emabf
        name, value = bf
        assert emaname == name, 'name not equal:{} , {}'.format(emaname, name)
        # if 'running_mean' in name or 'running_var' in name:
        if cfg.SELF_TRAIN.get('EMA_BN', None):
            # alpha = 0.95
            emavalue.data = emavalue.data * alpha + value.data * (1 - alpha)
        elif cfg.SELF_TRAIN.get('EMA_COPY', None):
            emavalue.data = value.data

def train_model_st(model, optimizer, source_loader, target_loader, model_func, lr_scheduler, optim_cfg,
                   start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, ps_label_dir,
                   source_sampler=None, target_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                   max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None, 
                   source_loader_detect=None, source_sampler_detect=None, source_model=None, dist=None):
    accumulated_iter = start_iter
    source_reader = common_utils.DataReader(source_loader, source_sampler)
    source_reader.construct_iter()

    # for continue training.
    # if already exist generated pseudo label result
    ps_pkl = self_training_utils.check_already_exsit_pseudo_label(ps_label_dir, start_epoch)
    if ps_pkl is not None:
        logger.info('==> Loading pseudo labels from {}'.format(ps_pkl))

    # for continue training
    if cfg.SELF_TRAIN.get('PROG_AUG', None) and cfg.SELF_TRAIN.PROG_AUG.ENABLED and \
        start_epoch > 0:
        for cur_epoch in range(start_epoch):
            if cur_epoch in cfg.SELF_TRAIN.PROG_AUG.UPDATE_AUG:
                target_loader.dataset.data_augmentor.re_prepare(
                    augmentor_configs=None, intensity=cfg.SELF_TRAIN.PROG_AUG.SCALE)

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True,
                     leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(target_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(target_loader.dataset, 'merge_all_iters_to_one_epoch')
            target_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(target_loader) // max(total_epochs, 1)

        dataloader_iter = iter(target_loader)

        for cur_epoch in tbar:
            if target_sampler is not None:
                target_sampler.set_epoch(cur_epoch)
                source_reader.set_cur_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # update pseudo label
            if (cur_epoch in cfg.SELF_TRAIN.UPDATE_PSEUDO_LABEL) or \
                    ((cur_epoch % cfg.SELF_TRAIN.UPDATE_PSEUDO_LABEL_INTERVAL == 0)
                     and cur_epoch != 0):
                target_loader.dataset.eval()
                self_training_utils.save_pseudo_label_epoch(
                    ema_model if ema_model else model, target_loader, rank,
                    leave_pbar=True, ps_label_dir=ps_label_dir, cur_epoch=cur_epoch)

                if cfg.SELF_TRAIN.get('OBJECT', None) and cfg.SELF_TRAIN.OBJECT.get('rdu', None):
                    cfg.SELF_TRAIN.OBJECT.max = cfg.SELF_TRAIN.OBJECT.max - 1

                
                if cfg.SELF_TRAIN.get('OBJECT', None) and cfg.SELF_TRAIN.OBJECT.get('increse', None):
                    cfg.SELF_TRAIN.OBJECT.max = cfg.SELF_TRAIN.OBJECT.max - 1


                target_loader.dataset.train()

                if cfg.SELF_TRAIN.get('ADABN_EPOCH', None):
                    model.train()
                    if ema_model is not None:
                        ema_model.train()
                    if cfg.SELF_TRAIN.get('DSNORM', None):
                        model.apply(set_ds_target)
                    if ema_model is not None:
                        ema_model.apply(set_ds_target)
                    for cur_it in range(total_it_each_epoch):
                        try:
                            target_batch = next(dataloader_iter)
                        except StopIteration:
                            dataloader_iter = iter(target_loader)
                            target_batch = next(dataloader_iter)
                            print('new iters')
                        with torch.no_grad():
                            load_data_to_gpu(target_batch)
                            _ = model(target_batch)[0]
                    if ema_model is not None:
                        update_ema_variables(model, ema_model, model_cfg=ema_model.model_cfg)
            
            # curriculum data augmentation
            if cfg.SELF_TRAIN.get('PROG_AUG', None) and cfg.SELF_TRAIN.PROG_AUG.ENABLED and \
                (cur_epoch in cfg.SELF_TRAIN.PROG_AUG.UPDATE_AUG):
                target_loader.dataset.data_augmentor.re_prepare(
                    augmentor_configs=None, intensity=cfg.SELF_TRAIN.PROG_AUG.SCALE)

            accumulated_iter = train_one_epoch_st(
                model, optimizer, source_reader, target_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter, ema_model=ema_model, cur_epoch=cur_epoch
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                state = checkpoint_state(model, optimizer, trained_epoch, accumulated_iter)

                save_checkpoint(state, filename=ckpt_name)
