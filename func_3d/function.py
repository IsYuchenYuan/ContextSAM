""" function for training and validation in one epoch
    Yunli Qi
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm

import cfg
from conf import settings
from func_3d.utils import eval_seg, ORGAN_NAME
import numpy as np
import ipdb
from collections import defaultdict
import json
import time

args = cfg.parse_args()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1, focal_weight=1):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(to_onehot_y=True, sigmoid=True)
        self.focal_loss = FocalLoss(to_onehot_y=True, gamma=2.0)

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.dice_weight * dice + self.focal_weight * focal


GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice) * 2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1, 11, (1, 7))

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []


def train_sam(args, net: nn.Module, optimizer1, optimizer2, train_loader, epoch, schedulers):
    epoch_loss = 0
    epoch_prompt_loss = 0
    epoch_non_prompt_loss = 0
    # train mode
    net.train()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    prompt = args.prompt
    lr_scale = args.lr_scale
    lr_ = args.base_lr
    lossfunc = criterion_G
    prompt_freq = args.prompt_freq

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for (batch_id, pack) in enumerate(train_loader):
            if pack == torch.zeros([1]):
                continue
            torch.cuda.empty_cache()
            imgs_tensor = pack['image']  # 1,d,3,h,w
            mask_dict = pack['label']  # {'frame_idx',{'obj',mask(1,h,w)}}
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']  # {'frame_idx',{'obj',coordinates(1,4)}}

            imgs_tensor = imgs_tensor.squeeze(0)  # T,3,h,w
            imgs_tensor = imgs_tensor.to(dtype=torch.float32, device=GPUdevice)
            num_frames = imgs_tensor.shape[0]
            train_state = net.train_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, num_frames, prompt_freq))
            obj_list = []
            for id in prompt_frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            name = pack['image_meta_dict']['filename_or_obj']

            with torch.cuda.amp.autocast():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ = net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),  # 1，4
                                    clear_old_points=False,
                                )
                        except KeyError:
                            print(f"no prompt input for cls: {ann_obj_id}")
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(train_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(train_state,
                                                                                                reverse=True):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                non_prompt_loss = 0
                prompt_loss = 0
                for id in range(num_frames):
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        obj_loss = lossfunc(pred, mask)
                        loss += obj_loss.item()
                        if id in prompt_frame_id:
                            prompt_loss += obj_loss
                        else:
                            non_prompt_loss += obj_loss
                loss = loss / num_frames / len(obj_list)
                if (num_frames - len(prompt_frame_id)) > 0:
                    non_prompt_loss = non_prompt_loss / (num_frames - len(prompt_frame_id)) / len(obj_list)
                prompt_loss = prompt_loss / len(prompt_frame_id) / len(obj_list)

                pbar.set_postfix(**{'loss (batch)': loss})
                epoch_loss += loss
                epoch_prompt_loss += prompt_loss.item()
                if isinstance(non_prompt_loss, torch.Tensor):
                    epoch_non_prompt_loss += non_prompt_loss.item()
                else:
                    epoch_non_prompt_loss += non_prompt_loss
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                if isinstance(non_prompt_loss, torch.Tensor) and optimizer2 is not None:
                    optimizer2.zero_grad()
                    non_prompt_loss.backward(retain_graph=True)
                    optimizer2.step()  # update memory related modules
                if optimizer1 is not None and prompt_loss.requires_grad:
                    optimizer1.zero_grad()
                    prompt_loss.backward()  # update mask decoder
                    optimizer1.step()

                for scheduler in schedulers:
                    scheduler.step()
                lr_ = optimizer1.param_groups[0]['lr']

                net.reset_state(train_state)

            pbar.update()

    return epoch_loss / len(train_loader), epoch_prompt_loss / len(train_loader), epoch_non_prompt_loss / len(
        train_loader), lr_


def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    n_val = len(val_loader)  # the number of batch
    mix_res = (0,) * 1 * 2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss

    prompt = args.prompt

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            if pack == torch.zeros([1]):
                continue
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            frame_id = list(range(imgs_tensor.size(0)))
            num_frame, _, h, w = imgs_tensor.size()
            train_state = net.val_init_state(imgs_tensor=imgs_tensor)

            prompt_id_inframe = pack['prompt_id_inframe'].squeeze(0).item()
            prompt_frame_id = [prompt_id_inframe]
            obj_list = pack['ann_obj_list']  # 1,1,num
            obj_list = obj_list.squeeze(0).squeeze(0).tolist()

            name = pack['image_meta_dict']['filename_or_obj']

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ = net.add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ = net.add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )

                video_segments = {}  # video_segments contains the per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(train_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(train_state,
                                                                                                reverse=True):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                pred_iou = 0
                pred_dice = 0
                for id in frame_id:
                    for (i, ann_obj_id) in enumerate(obj_list):
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        loss += lossfunc(pred, mask).item()
                        temp = eval_seg(pred, mask, threshold)
                        pred_iou += temp[0]
                        pred_dice += temp[1]

                total_num = len(frame_id) * len(obj_list)
                loss = loss / total_num
                temp = (pred_iou / total_num, pred_dice / total_num)
                tot += loss

                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            net.reset_state(train_state)
            pbar.update()

    return tuple([a / n_val for a in mix_res])


def simple_clip_and_monitor(parameters, max_norm, name=""):
    """简化版的梯度裁剪和监控"""
    # 计算原始梯度范数
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    original_norm = (total_norm ** 0.5)

    # 执行梯度裁剪
    clipped_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm).item()
    print(f"{name} - Gradient clipped: {original_norm:.4f} -> {clipped_norm:.4f} ")
    # 如果梯度被显著裁剪，打印信息
    if original_norm > max(clipped_norm, 1e-6) * 1.5:
        print(f"{name} - Gradient clipped: {original_norm:.4f} -> {clipped_norm:.4f} "
              f"(ratio: {clipped_norm / original_norm:.2%})")


def print_gradients(parameters, component_name):
    """
    打印模型梯度信息

    Args:
        model: 模型
        step: 当前步数
        component_name: 组件名称
    """
    total_norm = 0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f'{component_name} gradient norm: {total_norm:.4f}')


def train_sam_spt(args, net: nn.Module, optimizer, train_loader,
                  epoch, schedulers):
    epoch_loss = 0
    epoch_p_loss = 0
    epoch_sup_loss = 0
    epoch_moe_loss = 0
    epoch_moe_cls_loss = 0
    epoch_moe_blc_loss = 0
    lr_ = 0
    # train mode
    net.train()
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    prompt = args.prompt
    lossfunc = criterion_G
    for opt in optimizer:
        opt.zero_grad()
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        # start_time = time.time()
        for (batch_idx, pack) in enumerate(train_loader):
            # end_time = time.time()
            # print(f'load dataset time:{end_time-start_time}')
            if pack == torch.zeros([1]):
                continue
            organ_id = pack['organ_id'].squeeze(0).item()
            name = pack['image_meta_dict']['filename_or_obj']
            torch.cuda.empty_cache()
            imgs_tensor = pack['image']  # 1,T,3,h,w
            label_tensor = pack['label']  # # 1,T,h,w
            sup_img = pack['sup_img']  # 1,k,3,h,w: tensor
            sup_mask = pack['sup_mask']  # 1,k,1,h,w: tensor
            if len(imgs_tensor.shape) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)  # T,3,h,w
                label_tensor = label_tensor.squeeze(0)  # T,h,w
            num_frames = imgs_tensor.shape[0]
            if len(sup_img.shape) == 5:
                sup_img = sup_img.squeeze(0)
                sup_mask = sup_mask.squeeze(0)
            sup_num = sup_img.shape[0]
            contain_obj = pack['contain_obj']  # 1,d,1
            contain_obj = contain_obj.squeeze(0).to(dtype=torch.float32, device=GPUdevice)  # d,1

            text_prompt = f'A computerized tomography of a {ORGAN_NAME[organ_id - 1]}'
            print(text_prompt)
            # if batch_idx == 95:
            #     ipdb.set_trace()
            prompt_id_inframe = pack['prompt_id_inframe'].squeeze(0).item()
            prompt_frame_id = [prompt_id_inframe + sup_num]  # 多了一张support img

            #         import cv2
            #         name=name[0]
            #         imgs_tensor = imgs_tensor.cpu()
            #         label_tensor = label_tensor.cpu()
            #
            #         # 创建保存目录
            #         save_dir = "./datares"
            #         os.makedirs(save_dir, exist_ok=True)
            #
            #         # 遍历每个切片
            #         img = imgs_tensor[prompt_id_inframe].permute(1, 2, 0).numpy()  # 转换为(h,w,3)
            #         mask = label_tensor[prompt_id_inframe].numpy()  # (h,w)
            #
            #         # 将图像转换为0-255范围
            #         img = (img).astype(np.uint8)
            #
            #         # 创建红色覆盖层
            #         red_mask = np.zeros_like(img)
            #         red_mask[mask > 0] = [0, 0, 255]  # BGR格式
            #
            #         # 叠加图像（原图70%，mask30%）
            #         overlay = cv2.addWeighted(img, 0.7, red_mask, 0.3, 0)
            #
            #         # 保存图像
            #         if name.split('/')[0] == '10_Decathlon':
            #             name = name.split('/')[0] + '_' + name.split('/')[1]
            #         else:
            #             name = name.split('/')[0]
            #         save_path = os.path.join(save_dir, f"{name}_{ORGAN_NAME[organ_id - 1]}_slice_{prompt_id_inframe:03d}.png")
            #         cv2.imwrite(save_path, overlay)
            #
            #         print(f"已保存{imgs_tensor.shape[0]}张切片图像到{save_dir}目录")
            #         continue
            # assert 1==0
            """-------------------------version 1108--------------------------"""
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']  # {'frame_idx',coordinates(1,4)}}

            all_imgs = torch.cat((sup_img, imgs_tensor), dim=0)  # 1+T,3,h,w
            train_state = net.train_init_state(imgs_tensor=all_imgs, sup_num=sup_num, mask_dict=label_tensor,
                                               text_prompt=text_prompt)

            with torch.cuda.amp.autocast():
                # support image as conditional slice
                for sup_id in range(sup_num):
                    sup_mask_obj = sup_mask[sup_id].clone()
                    _, _, _ = net.train_add_new_mask(
                        inference_state=train_state,
                        frame_idx=sup_id,
                        obj_id=1,  # 每个标签都是binary的
                        mask=sup_mask_obj.squeeze(),
                    )
                for id in prompt_frame_id:
                    try:
                        if prompt == 'click':
                            points = pt_dict[id - sup_num].to(device=GPUdevice)
                            labels = point_labels_dict[id - sup_num].to(device=GPUdevice)
                            _, _, _ = net.train_add_new_points(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=1,
                                points=points,
                                labels=labels,
                                clear_old_points=False,
                            )
                        elif prompt == 'bbox':
                            bbox = bbox_dict[id - sup_num]
                            _, _, _ = net.train_add_new_bbox(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=1,
                                bbox=bbox.to(device=GPUdevice),  # 1，4
                                clear_old_points=False,
                            )
                    except KeyError:
                        print(f"no prompt input for the object organ")
                        _, _, _ = net.train_add_new_mask(
                            inference_state=train_state,
                            frame_idx=id,
                            obj_id=1,
                            mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                        )
                video_segments = {}  # video_segments contains the per-frame segmentation results
                video_segments_bysup = {}  # video_segments produced by support image contains the per-frame segmentation results
                video_moe_loss = 0
                video_moe_cls_loss = 0
                video_moe_blc_loss = 0
                for out_frame_idx, out_obj_ids, out_mask_logits, sup_out_mask_logits, obj_logit, obj_logit_sup, (
                        moe_loss, moe_cls_loss, moe_blc_loss) in net.train_propagate_in_video(
                    train_state, start_frame_idx=prompt_frame_id[0]):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    video_segments_bysup[out_frame_idx] = {
                        out_obj_id: sup_out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    video_moe_loss += moe_loss
                    video_moe_cls_loss += moe_cls_loss
                    video_moe_blc_loss += moe_blc_loss

                for out_frame_idx, out_obj_ids, out_mask_logits, sup_out_mask_logits, obj_logit, obj_logit_sup, (
                        moe_loss, moe_cls_loss, moe_blc_loss) in net.train_propagate_in_video(
                    train_state, start_frame_idx=prompt_frame_id[0], reverse=True):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    video_segments_bysup[out_frame_idx] = {
                        out_obj_id: sup_out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    video_moe_loss += moe_loss
                    video_moe_cls_loss += moe_cls_loss
                    video_moe_blc_loss += moe_blc_loss

                    # allocated_memory = torch.cuda.memory_allocated()
                    # print(f"Allocated memory after reset_state: {allocated_memory / 1024 ** 2:.2f} MB")

                p_loss = 0
                prompt_loss = 0
                non_prompt_loss = 0
                sup_loss = 0

                for id in range(sup_num, num_frames + sup_num):
                    pred = video_segments[id][1]
                    pred = pred.unsqueeze(0)
                    pred_bysup = video_segments_bysup[id][1]
                    pred_bysup = pred_bysup.unsqueeze(0)
                    mask = label_tensor[id - sup_num].to(dtype=torch.float32, device=GPUdevice)
                    assert torch.max(mask) != 255
                    mask = mask.unsqueeze(0).unsqueeze(0)
                    if args.train_vis and epoch % args.train_vis_epoch == 0:
                        os.makedirs(f'./temp/train/{args.exp_name}/{epoch}/{name[0]}/{id}', exist_ok=True)
                        fig, ax = plt.subplots(1, 4)
                        ax[0].imshow(all_imgs[id, :, :, :].detach().cpu().permute(1, 2, 0).numpy().astype(int))
                        ax[0].axis('off')
                        ax[1].imshow(pred[0, 0, :, :].detach().cpu().numpy() > 0, cmap='gray')
                        ax[1].axis('off')
                        ax[2].imshow(pred_bysup[0, 0, :, :].detach().cpu().numpy() > 0, cmap='gray')
                        ax[2].axis('off')
                        try:
                            bbox = bbox_dict[id - sup_num]
                            ax[1].add_patch(plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0],
                                                          bbox[0][3] - bbox[0][1], edgecolor='green',
                                                          facecolor=(0, 0, 0, 0), lw=2))
                        except KeyError:
                            pass
                        ax[3].imshow(mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                        ax[3].axis('off')
                        plt.savefig(
                            f'./temp/train/{args.exp_name}/{epoch}/{name[0]}/{id}/temp.png',
                            bbox_inches='tight', pad_inches=0)
                        plt.close()

                    obj_loss = lossfunc(pred, mask)
                    dice = eval_seg(pred, mask, (0,))
                    print(f'dice:{dice[0]}')
                    if id in prompt_frame_id:
                        prompt_loss += obj_loss
                    else:
                        non_prompt_loss += obj_loss
                    sup_obj_loss = lossfunc(pred_bysup, mask)
                    p_loss += obj_loss
                    sup_loss += sup_obj_loss

                sup_loss = sup_loss / num_frames  # matching
                p_loss = p_loss / num_frames  # propagation
                video_moe_loss = video_moe_loss / num_frames
                video_moe_cls_loss = video_moe_cls_loss / num_frames
                video_moe_blc_loss = video_moe_blc_loss / num_frames
                prompt_loss = prompt_loss / len(prompt_frame_id)
                loss = p_loss + sup_loss + 0.1 * video_moe_loss

                moe_loss_value = video_moe_loss.item() if isinstance(video_moe_loss, torch.Tensor) else video_moe_loss
                moe_cls_loss_value = video_moe_cls_loss.item() if isinstance(video_moe_cls_loss,
                                                                             torch.Tensor) else video_moe_cls_loss
                moe_blc_loss_value = video_moe_blc_loss.item() if isinstance(video_moe_blc_loss,
                                                                             torch.Tensor) else video_moe_blc_loss
                print(f'loss:{loss.item():.4f},p_loss: {p_loss.item():.4f}, '
                      f'sup_loss: {sup_loss.item():.4f}, moe_loss: {moe_loss_value:.4f}, '
                      f'moe_cls_loss: {moe_cls_loss_value:.4f}, moe_blc_loss: {moe_blc_loss_value:.4f}')
                epoch_loss += loss.item()
                epoch_p_loss += p_loss.item()
                epoch_sup_loss += sup_loss.item()
                epoch_moe_loss += moe_loss_value
                epoch_moe_cls_loss += moe_cls_loss_value
                epoch_moe_blc_loss += moe_blc_loss_value

                if len(optimizer) == 2:
                    optimizer1, optimizer2 = optimizer[0], optimizer[1]
                    if optimizer2 is not None:  # memory+encoder(lora)
                        loss.backward(retain_graph=True)
                        # 分别处理memory layers和encoder parameters
                        mem_params = optimizer2.param_groups[0]['params']
                        print_gradients(mem_params, 'Memory')
                        if args.lora_type in ['moe', 'single']:
                            encoder_params = optimizer2.param_groups[1]['params']
                            print_gradients(encoder_params, 'encoder')
                        optimizer2.step()
                        optimizer2.zero_grad()
                    if optimizer1 is not None:
                        prompt_loss.backward()  # update mask decoder
                        decoder_params = optimizer1.param_groups[0]['params']
                        print_gradients(decoder_params, "Mask Decoder")
                        optimizer1.step()
                        optimizer1.zero_grad()
                elif len(optimizer) == 1:
                    optimizer1 = optimizer[0]
                    loss.backward()  # update mask decoder
                    decoder_params = optimizer1.param_groups[0]['params']
                    mem_params = optimizer1.param_groups[1]['params']
                    encoder_params = optimizer1.param_groups[2]['params']
                    print_gradients(decoder_params, "Mask Decoder")
                    print_gradients(mem_params, 'Memory')
                    print_gradients(encoder_params, 'encoder')
                    optimizer1.step()
                    optimizer1.zero_grad()

                for scheduler in schedulers:
                    if scheduler is not None:
                        scheduler.step()

                # 在训练循环中
                # if optimizer2 is not None:  # memory+encoder(lora)
                #     loss.backward(retain_graph=True)
                #
                #     # 分别处理memory layers和encoder parameters
                #     mem_params = optimizer2.param_groups[0]['params']
                #     encoder_params = optimizer2.param_groups[1]['params']
                #
                #     # 对memory layers进行梯度裁剪和监控
                #     simple_clip_and_monitor(mem_params, max_norm=1.0, name="Memory Layers")
                #
                #     # 对encoder parameters进行梯度裁剪和监控
                #     simple_clip_and_monitor(encoder_params, max_norm=1.0, name="Encoder Parameters")
                #
                #     optimizer2.step()
                #     optimizer2.zero_grad()
                #
                # if optimizer1 is not None:
                #     prompt_loss.backward()  # update mask decoder
                #
                #     # 对mask decoder进行梯度裁剪和监控
                #     decoder_params = optimizer1.param_groups[0]['params']
                #     simple_clip_and_monitor(decoder_params, max_norm=1.0, name="Mask Decoder")
                #
                #     optimizer1.step()
                #     optimizer1.zero_grad()
                #
                # for scheduler in schedulers:
                #     if scheduler is not None:
                #         scheduler.step()

                lr_ = optimizer1.param_groups[0]['lr']
                net.reset_state(train_state)

            pbar.update()
    return epoch_loss / len(train_loader), epoch_p_loss / len(train_loader), \
           epoch_sup_loss / len(train_loader), epoch_moe_loss / len(train_loader), epoch_moe_cls_loss / len(
        train_loader), epoch_moe_blc_loss / len(train_loader), lr_


def validation_sam_spt(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()
    n_val = len(val_loader)  # the number of batch
    threshold = (0,)

    prompt = args.prompt
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, pack in enumerate(val_loader):
            if pack == torch.zeros([1]):
                continue
            imgs_tensor = pack['image']  # 1,T,3,h,w
            label_tensor = pack['label']  # # 1,T,h,w
            sup_img = pack['sup_img']  # 1,k,3,h,w: tensor
            sup_mask = pack['sup_mask']  # 1,k,1,h,w: tensor
            if len(imgs_tensor.shape) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)  # T,3,h,w
                label_tensor = label_tensor.squeeze(0)  # T,h,w
            num_frames = imgs_tensor.shape[0]
            if len(sup_img.shape) == 5:
                sup_img = sup_img.squeeze(0)
                sup_mask = sup_mask.squeeze(0)
            sup_num = sup_img.shape[0]
            name = pack['image_meta_dict']['filename_or_obj']
            organ_id = pack['organ_id'].squeeze(0).item()
            text_prompt = f'A computerized tomography of a {ORGAN_NAME[organ_id - 1]}'
            print(text_prompt)
            prompt_id_inframe = pack['prompt_id_inframe'].squeeze(0).item()
            prompt_frame_id = [prompt_id_inframe + sup_num]  # 多了一张support img

            all_imgs = torch.cat((sup_img, imgs_tensor), dim=0)  # 1+T,3,h,w
            eval_state = net.val_init_state(imgs_tensor=all_imgs, sup_num=sup_num, text_prompt=text_prompt)

            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']  # {'frame_idx',coordinates(1,4)}}

            with torch.no_grad():
                for sup_id in range(sup_num):
                    sup_mask_obj = sup_mask[sup_id].clone()
                    _, _, _ = net.train_add_new_mask(
                        inference_state=eval_state,
                        frame_idx=sup_id,
                        obj_id=1,
                        mask=sup_mask_obj.squeeze(),
                    )

                for id in prompt_frame_id:
                    try:
                        if prompt == 'click':
                            points = pt_dict[id - sup_num].to(device=GPUdevice)
                            labels = point_labels_dict[id - sup_num].to(device=GPUdevice)
                            _, _, _ = net.add_new_points(
                                inference_state=eval_state,
                                frame_idx=id,
                                obj_id=1,
                                points=points,
                                labels=labels,
                                clear_old_points=False,
                            )
                        elif prompt == 'bbox':
                            bbox = bbox_dict[id - sup_num]
                            _, _, _ = net.add_new_bbox(
                                inference_state=eval_state,
                                frame_idx=id,
                                obj_id=1,
                                bbox=bbox.to(device=GPUdevice),  # 1，4
                                clear_old_points=False,
                            )
                    except KeyError:
                        _, _, _ = net.add_new_mask(
                            inference_state=eval_state,
                            frame_idx=id,
                            obj_id=1,
                            mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                        )

                video_segments = {}  # video_segments contains the per-frame segmentation results
                video_segments_bysup = {}  # video_segments produced by support image contains the per-frame segmentation results

                for out_frame_idx, out_obj_ids, out_mask_logits, sup_out_mask_logits, _, _ in net.propagate_in_video(
                        eval_state,
                        start_frame_idx=
                        prompt_frame_id[0]):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    video_segments_bysup[out_frame_idx] = {
                        out_obj_id: sup_out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                for out_frame_idx, out_obj_ids, out_mask_logits, sup_out_mask_logits, _, _ in net.propagate_in_video(
                        eval_state,
                        start_frame_idx=
                        prompt_frame_id[0],
                        reverse=True):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    video_segments_bysup[out_frame_idx] = {
                        out_obj_id: sup_out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                p_dice_byorgan = defaultdict(list)
                m_dice_byorgan = defaultdict(list)
                p_pred_dice = 0
                m_pred_dice = 0
                for id in range(sup_num, num_frames + sup_num):
                    pred = video_segments[id][1]
                    pred = pred.unsqueeze(0)
                    pred_bysup = video_segments_bysup[id][1]
                    pred_bysup = pred_bysup.unsqueeze(0)
                    mask = label_tensor[id - sup_num].to(dtype=torch.float32, device=GPUdevice)
                    assert torch.max(mask) != 255
                    mask = mask.unsqueeze(0).unsqueeze(0)
                    # if args.vis and epoch % args.val_freq == 0:
                    #     os.makedirs(f'./temp/eval/{args.exp_name}/{epoch}/{name[0]}/{id}', exist_ok=True)
                    #     fig, ax = plt.subplots(1, 4)
                    #     ax[0].imshow(all_imgs[id, :, :, :].detach().cpu().permute(1, 2, 0).numpy().astype(int))
                    #     ax[0].axis('off')
                    #     ax[1].imshow(pred[0, 0, :, :].detach().cpu().numpy() > 0, cmap='gray')
                    #     ax[1].axis('off')
                    #     ax[2].imshow(pred_bysup[0, 0, :, :].detach().cpu().numpy() > 0, cmap='gray')
                    #     ax[2].axis('off')
                    #     try:
                    #         bbox = bbox_dict[id - sup_num][ann_obj_id]
                    #         ax[1].add_patch(plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0],
                    #                                       bbox[0][3] - bbox[0][1], edgecolor='green',
                    #                                       facecolor=(0, 0, 0, 0), lw=2))
                    #     except KeyError:
                    #         pass
                    #     ax[3].imshow(mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                    #     ax[3].axis('off')
                    #     plt.savefig(
                    #         f'./temp/eval/{args.exp_name}/{epoch}/{name[0]}/{id}/{ann_obj_list.index(ann_obj_id)}.png',
                    #         bbox_inches='tight', pad_inches=0)
                    #     plt.close()

                    propagate_res = eval_seg(pred, mask, threshold)
                    matching_res = eval_seg(pred_bysup, mask, threshold)
                    p_pred_dice += propagate_res[1]
                    m_pred_dice += matching_res[1]

                total_num = num_frames
                p_pred_dice = p_pred_dice / total_num
                m_pred_dice = m_pred_dice / total_num
                print(f'p_pred_dice:{p_pred_dice},m_pred_dice:{m_pred_dice}')

                p_dice_byorgan[ORGAN_NAME[organ_id - 1]].append(p_pred_dice)
                m_dice_byorgan[ORGAN_NAME[organ_id - 1]].append(m_pred_dice)

            net.reset_state(eval_state)
            pbar.update()

    p_dice_byorgan_means = {k: np.mean(v) for k, v in p_dice_byorgan.items()}
    mix_res_p = np.mean(list(p_dice_byorgan_means.values()))
    m_dice_byorgan_means = {k: np.mean(v) for k, v in m_dice_byorgan.items()}
    mix_res_m = np.mean(list(m_dice_byorgan_means.values()))

    saveres = {}
    saveres['p_dice_byorgan'] = p_dice_byorgan_means
    saveres['m_dice_byorgan'] = m_dice_byorgan_means
    save_dir_path = f'./valres/{args.exp_name}_loralr_{args.lora_lr}_declr_{args.dec_lr}_memlr_{args.mem_lr}'
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    save_path = f'{save_dir_path}/{epoch}_res.json'
    with open(save_path, 'w') as f:
        json.dump(saveres, f, indent=2)

    return mix_res_p, mix_res_m
