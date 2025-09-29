# train.py
# !/usr/bin/env	python3

import os
import time

import torch
import cfg
import torch.backends.cudnn as cudnn
import random
import numpy as np
from func_3d import function
from func_3d.utils import get_network, set_log_dir, create_logger, calculate_metric_percase, random_click, \
    generate_bbox, RobustBoxMapper, TEMPLATE, ORGAN_NAME, augment_data, dataset_organname
from func_3d.dataset import get_dataloader_test

from tqdm import tqdm
import nibabel as nib
import numpy as np
from PIL import Image
import cv2
from collections import defaultdict
import json
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch.cuda.amp import autocast


dataset_clsNum = {
    'T2SPIR': [range(1, 5), './data/Test_organ/CHAOS_MRI', ['style', 'shape']],
    'T1DUAL_InPhase': [range(1, 5), './data/Test_organ/CHAOS_MRI', ['style', 'shape']],
    'T1DUAL_OutPhase': [range(1, 5), './data/Test_organ/CHAOS_MRI', ['style', 'shape']],
    'acdc': [range(1, 4), './data/Test_organ/acdc', ['style', 'shape']],  # 'style', 'shape', 'flip'
    'ski10': [range(1, 5), './data/Test_organ/ski10', ['style', 'shape']],
    'camus_2ch': [range(1, 4), './data/Test_organ/camus_2ch', ['style']],
    'camus_4ch': [range(1, 4), './data/Test_organ/camus_4ch', ['style']],
    'mmwhs': [range(1,8), './data/Test_organ/mmwhs', []],
    'word': [range(1, 18), './data/Test_organ/word', ['style', 'shape']],
}



colors = [
    (255, 0, 0),  # blue
    (0, 255, 0),  # green
    (0, 0, 255),  # red
    (255, 255, 0),  # 黄色
    (255, 0, 255),  # 洋红
    (0, 255, 255),  # 青色
    (255, 128, 0),  # 橙色
    (128, 0, 255),  # 紫色
    (0, 255, 128),  # 青绿色
    (255, 128, 128),  # 粉色
    (128, 255, 0),  # 黄绿色
    (255, 0, 128),  # 玫红色
    (128, 128, 255),  # 淡蓝色
    (255, 255, 128),  # 淡黄色
    (255, 128, 255),  # 淡紫色
    (128, 255, 255),  # 淡青色
    (255, 69, 0),  # 红橙色
    (138, 43, 226),  # 紫罗兰色
    (0, 206, 209),  # 深青色
    (255, 215, 0),  # 金色
    (50, 205, 50),  # 酸橙绿
    (220, 20, 60)  # 猩红色
]


def normalize_img(img):
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def calculate_dice(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0
    return (2.0 * intersection) / union


def main():
    args = cfg.parse_args()
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    GPUdevice = torch.device('cuda', args.gpu_device)
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    net.to(dtype=torch.bfloat16)
    args.div = 0.4  # 选择slice的时候的diversity系数
    args.m_W = 0.4  # weight of matching result
    args.sup_vol_num = 3 # 这个参数只在做sup_vol_num的ablation实验中有用
    for dataset in list(dataset_clsNum.keys()):
        args.dataset = dataset
        args.data_path = dataset_clsNum[dataset][1]
        Evaluate(args, net)
        


def compute_dice(mask1, mask2):
    intersection = (mask1 & mask2).sum()
    union = mask1.sum() + mask2.sum()
    if union == 0:
        return 1
    return 2 * intersection / union


def select_most_similar(features, query_feature, topk):
    features = features.cpu().numpy()
    query_feature = query_feature.cpu().numpy()
    query_similarity = cosine_similarity(features, query_feature.reshape(1, -1)).flatten()
    top_k_indices = np.argsort(query_similarity)[-topk:][::-1]
    print(top_k_indices)
    return top_k_indices.tolist()


def select_diverse_and_relevant_features_multi_queries(features, query_features, num_features=10, alpha=0.4):
    """
    从给定的图像特征中筛选出与多个 query_features 相似且差异性最大的特征。

    参数：
        features (numpy.ndarray): 形状为 (n_samples, n_features) 的数组，表示候选特征。
        query_features (numpy.ndarray): 形状为 (n_queries, n_features) 的数组，表示多个查询特征。
        num_features (int): 最终保留的特征数量。
        alpha (float): 平衡 diversity 和 relevance 的权重因子（0~1之间）。

    返回：
        selected_indices (list): 被选择的特征索引列表。
    """
    if len(features) <= num_features:
        return list(range(len(features)))

    # 转换为numpy数组并计算相似度
    features = features.cpu().numpy()
    query_features = query_features.cpu().numpy()

    # 计算特征间的相似度矩阵
    similarity_matrix = cosine_similarity(features)

    # 计算与所有query features的相似度矩阵
    # shape: (n_samples, n_queries)
    query_similarity_matrix = cosine_similarity(features, query_features)

    # 计算每个候选特征与query features的综合相似度得分
    # 使用最大值而不是平均值，确保选择的特征至少与某个query feature高度相关
    query_relevance_scores = np.max(query_similarity_matrix, axis=1)

    # 初始化：选择与query features最相关的特征作为起点
    initial_index = np.argmax(query_relevance_scores)
    selected_indices = [initial_index]
    remaining_indices = list(range(len(features)))
    remaining_indices.remove(initial_index)

    while len(selected_indices) < num_features:
        max_score = -np.inf
        best_index = None

        for idx in remaining_indices:
            # 1. 计算diversity得分（与已选特征的差异度）
            diversity_score = -np.mean([similarity_matrix[idx][sel_idx] for sel_idx in selected_indices])

            # 2. 计算relevance得分（与query features的相关度）
            # 使用当前特征对所有query features的覆盖程度
            current_relevance = query_similarity_matrix[idx]

            # 计算已选特征集合对每个query feature的最大相似度
            selected_relevance = np.max(query_similarity_matrix[selected_indices], axis=0)

            # 计算当前特征能否提供额外的覆盖
            combined_relevance = np.maximum(selected_relevance, current_relevance)
            relevance_improvement = np.mean(combined_relevance) - np.mean(selected_relevance)

            # 3. 综合得分
            combined_score = alpha * diversity_score + (1 - alpha) * relevance_improvement

            if combined_score > max_score:
                max_score = combined_score
                best_index = idx

        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    return selected_indices


def select_diverse_and_relevant_features(features, query_feature, num_features=10, alpha=0.4):
    """
    从给定的图像特征中筛选出与 query_feature 相似且差异性最大的特征。

    参数：
        features (numpy.ndarray): 一个形状为 (n_samples, n_features) 的数组，表示 n 个图像的特征。
        query_feature (numpy.ndarray): 一个形状为 (1, n_features) 的数组，表示查询特征。
        num_features (int): 最终保留的特征数量。
        alpha (float): 平衡 diversity 和与 query_feature 相似度的权重因子（0~1之间）。越大越偏向 diversity。

    返回：
        selected_indices (list): 被选择的特征索引列表。
        selected_features (numpy.ndarray): 被选择的特征数组，形状为 (num_features, n_features)。
    """
    if len(features) <= num_features:
        return list(range(len(features))), features

    # Step 1: 计算特征之间的相似度矩阵
    features = features.cpu().numpy()
    query_feature = query_feature.cpu().numpy()
    similarity_matrix = cosine_similarity(features)
    query_similarity = cosine_similarity(features, query_feature.reshape(1, -1)).flatten()

    # Step 2: 初始化选择结果，随机选择一个特征作为初始点
    initial_index = np.random.choice(len(features))  # 随机选择一个索引
    selected_indices = [initial_index]
    remaining_indices = list(range(len(features)))
    remaining_indices.remove(initial_index)

    while len(selected_indices) < num_features:
        max_score = -np.inf
        best_index = None

        for idx in remaining_indices:
            # 计算与 query_feature 的相似度得分
            relevance_score = query_similarity[idx]

            # 计算与已选特征的 diversity 得分（负的平均相似度）
            diversity_score = -np.mean([similarity_matrix[idx][sel_idx] for sel_idx in selected_indices])

            # 综合得分：alpha 控制 diversity 和 relevance 的权重
            combined_score = alpha * diversity_score + (1 - alpha) * relevance_score

            if combined_score > max_score:
                max_score = combined_score
                best_index = idx

        # 添加选择的特征索引
        selected_indices.append(best_index)
        remaining_indices.remove(best_index)
    print(selected_indices)
    return selected_indices


def select_diverse_features(features, sup_mid_stack_indices, num_features=10):
    """
    从给定的图像特征中选出差异性最大的特征。

    参数：
        features (numpy.ndarray): 一个形状为 (n_samples, n_features) 的数组，表示 n 个图像的特征。
        num_features (int): 最终保留的特征数量。

    返回：
        selected_indices (list): 被选择的特征索引列表。
        selected_features (numpy.ndarray): 被选择的特征数组，形状为 (num_features, n_features)。
    """
    if len(features) <= num_features:
        return list(range(len(features)))

    # 计算特征之间的相似度矩阵
    features = features.cpu().numpy()
    similarity_matrix = cosine_similarity(features)

    selected_indices = sup_mid_stack_indices
    remaining_indices = list(range(len(features)))

    for idx in selected_indices:
        remaining_indices.remove(idx)

    while len(selected_indices) < num_features:
        max_score = -np.inf
        best_index = None

        for idx in remaining_indices:
            # 计算与已选特征的差异性得分（负的平均相似度）
            diversity_score = -np.mean([similarity_matrix[idx][sel_idx] for sel_idx in selected_indices])

            if diversity_score > max_score:
                max_score = diversity_score
                best_index = idx
        # 添加选择的特征索引
        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    return selected_indices


def merge_features(feat):
    # 获取三个level的特征
    feat0 = feat['backbone_fpn'][0]  # [1, 32, 128, 128]
    feat1 = feat['backbone_fpn'][1]  # [1, 64, 64, 64]
    feat2 = feat['backbone_fpn'][2]  # [1, 256, 32, 32]

    # 将feat1和feat0调整到相同大小
    feat1_up = F.interpolate(feat1, size=feat0.shape[2:], mode='bilinear', align_corners=False)
    feat2_up = F.interpolate(feat2, size=feat0.shape[2:], mode='bilinear', align_corners=False)

    # 在channel维度上拼接
    merged = torch.cat([feat0, feat1_up, feat2_up], dim=1)  # [1, 352, 128, 128]

    # 将特征展平并归一化
    flat_feat = merged.view(merged.size(0), -1)  # [1, 352*128*128]
    norm_feat = F.normalize(flat_feat, p=2, dim=1)  # L2归一化
    return norm_feat


def Evaluate(args, net):
    net.eval()
    dataset = args.dataset
    test_save_path = os.path.join(args.test_save_path, args.net, dataset,
                                  f'supNum_{args.sup_num}_promptNum_{args.prompt_num}_promptType_{args.test_prompt}')
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    all_organ_avg = []
    all_organ_std = []
    all_organ = []

    organ_list = [idx - 1 for idx in list(dataset_clsNum[args.dataset][0])]

    for organ_idx, (organ, organ_name) in enumerate(zip(dataset_clsNum[args.dataset][0],
                                 [dataset_organname[args.dataset][i] for i in organ_list])):
        organ_dice = defaultdict(list)
        nice_test_loader = get_dataloader_test(args, organ)
        n_val = len(nice_test_loader)  # the number of batch
        prompt = args.test_prompt
        with autocast(dtype=torch.bfloat16):
            with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
                
                for (batch_idx, pack) in enumerate(nice_test_loader):
                    if args.test_prompt == 'None':
                        tmp_names = pack['tmp_names']
                        print(tmp_names)
                    else:
                        tmp_names = []
                    imgs_tensor = pack['image']
                    mask_tensor = pack['label']
                    sup_imgs = pack['sup_imgs']  # 1,k,c,h,w
                    sup_msks = pack['sup_msks']  # 1,k,c,h,w
                    sup_mid_stack_indices = pack['sup_mid_stack_indices']
                    sup_mid_stack_indices = [idx.item() for idx in sup_mid_stack_indices]
                    name = pack['image_meta_dict']['filename_or_obj'][0]
                    if prompt == 'click':
                        pt_dict = pack['pt']
                        point_labels_dict = pack['p_label']
                    elif prompt == 'bbox':
                        bbox_dict = pack['bbox']
                    if len(imgs_tensor.size()) == 5:
                        imgs_tensor = imgs_tensor.squeeze(0)  # d,3,h,w
                        mask_tensor = mask_tensor.squeeze(0)  # d,1,h,w
                        sup_imgs = sup_imgs.squeeze(0)  # k,c,h,w
                        sup_msks = sup_msks.squeeze(0)  # k,c,h,w

                    if batch_idx == 0:
                        print(sup_imgs.shape)
                        
                    num_frames = imgs_tensor.shape[0]
                    # case不存在这个类别，直接略过
                    need_infer = True
                    obj_ranges = pack['obj_ranges'][0].tolist()
                    if obj_ranges[0] is None or obj_ranges[0] == -1:
                        need_infer = False

                    text_prompt = f'A computerized tomography of a {organ}_{organ_name}'
                    print(text_prompt)
                    numframes = sup_imgs.shape[0]
                    if numframes >350:
                        mid_sup_imgs = sup_imgs[sup_mid_stack_indices]
                        mid_sup_msks = sup_msks[sup_mid_stack_indices]
                        remain_indices = [i for i in range(numframes) if i not in sup_mid_stack_indices]
                        remain_sup_imgs = sup_imgs[remain_indices][::2]  # k,c,h,w
                        remain_sup_msks = sup_msks[remain_indices][::2]   # k,c,h,w
                        sup_imgs = torch.cat((mid_sup_imgs,remain_sup_imgs),dim=0)
                        sup_msks = torch.cat((mid_sup_msks,remain_sup_msks),dim=0)
                        sup_mid_stack_indices = [0,1,2]
                    with torch.no_grad():
                        if need_infer:
                            # step1: select 10 sharing support slices based on diversity
                            sup_num = sup_imgs.shape[0]
                            cal_sim_state = net.val_init_state(imgs_tensor=sup_imgs, sup_num=sup_num,
                                                               process_sup=batch_idx == 0)
                            features = cal_sim_state["cached_features"]  # {frame_idx: (image, backbone_out)}
                            if batch_idx == 0:
                                support_features = []
                                for i in range(sup_num):
                                    support_merged = merge_features(features[i][1])
                                    support_features.append(support_merged)
                                support_features = torch.cat(support_features, dim=0)
                                select_sup_idx_list = select_diverse_features(support_features, 
                                                                              sup_mid_stack_indices=sup_mid_stack_indices,
                                                                              num_features=args.sup_num)

                            print(select_sup_idx_list)
                            sup_imgs = sup_imgs[select_sup_idx_list]  # k,c,h,w
                            sup_msks = sup_msks[select_sup_idx_list]  # k,1,h,w

                           # do the augmentation for the support images
                            aug = dataset_clsNum[args.dataset][2]
                            sup_imgs, sup_msks = augment_data(sup_imgs, sup_msks, aug=aug)
                            sup_num = sup_imgs.shape[0]
                            print(f"support number after augmentation: {sup_num}")
        
                            # step2: support-query matching
                            imgs_tensor = imgs_tensor[obj_ranges[0]:obj_ranges[1] + 1]
                            mask_tensor = mask_tensor[obj_ranges[0]:obj_ranges[1] + 1]
                            all_images = torch.cat((sup_imgs, imgs_tensor), dim=0)  # d+k,c,h,w
                            mid = (obj_ranges[1] + 1 - obj_ranges[0]) // 2
                            start_slice = mid + sup_num
                            eval_state = net.val_init_state(imgs_tensor=all_images, sup_num=sup_num)
                            for i in range(sup_num):
                                _, out_obj_ids, out_mask_logits = net.add_new_mask(
                                    inference_state=eval_state,
                                    frame_idx=i,
                                    obj_id=1,
                                    mask=(sup_msks[i][0] == 1).int(),
                                )

                            """---------------------use query prediction as prompt---------------------"""
                            if prompt == 'None':
                                print('Automatically select query slice.........')
                                video_segments = {}  # video_segments contains the per-frame segmentation results
                                video_segments_bysup = {}  # video_segments produced by support image contains the per-frame segmentation results
                                ious_sup_dict = {}

                                for out_frame_idx, out_obj_ids, out_mask_logits, sup_out_mask_logits, obj_logit, obj_logit_sup, ious_prop, ious_sup in net.propagate_in_video(
                                        eval_state,
                                        start_frame_idx=start_slice):
                                    video_segments[out_frame_idx] = {
                                        out_obj_id: out_mask_logits[i]
                                        for i, out_obj_id in enumerate(out_obj_ids)
                                    }
                                    video_segments_bysup[out_frame_idx] = {
                                        out_obj_id: sup_out_mask_logits[i]
                                        for i, out_obj_id in enumerate(out_obj_ids)
                                    }
                                    ious_sup_dict[out_frame_idx] = ious_sup

                                for out_frame_idx, out_obj_ids, out_mask_logits, sup_out_mask_logits, obj_logit, obj_logit_sup, ious_prop, ious_sup in net.propagate_in_video(
                                        eval_state,
                                        start_frame_idx=start_slice - 1,
                                        reverse=True):
                                    video_segments[out_frame_idx] = {
                                        out_obj_id: out_mask_logits[i]
                                        for i, out_obj_id in enumerate(out_obj_ids)
                                    }
                                    video_segments_bysup[out_frame_idx] = {
                                        out_obj_id: sup_out_mask_logits[i]
                                        for i, out_obj_id in enumerate(out_obj_ids)
                                    }
                                    ious_sup_dict[out_frame_idx] = ious_sup  # B,1
                                ious_sup_dict = {k: v for k, v in ious_sup_dict.items() if k >= sup_num}
                                
                                sorted_dict = dict(
                                    sorted(ious_sup_dict.items(), key=lambda item: torch.max(item[1]),
                                           reverse=True))
                                num = min(7, len(list(sorted_dict.keys())))
                                slc_ids = list(sorted_dict.keys())[:num]
                                start_slice = min(slc_ids, key=lambda x: abs(x - mid - sup_num))
                              
                                
                                """---------------------可视化query选择结果---------------------"""
                                if args.save_result:
                                    imgs_tensor_np = all_images.cpu().numpy()
                                    all_vis = []
                                    img_vis = []
                                    for sup_idx in range(sup_imgs.shape[0]):
                                        spt_img = sup_imgs[sup_idx].squeeze(0).permute(1, 2, 0).cpu().numpy().astype(
                                            np.uint8)
                                        img_vis.append(spt_img)
                                        spt_mask = sup_msks[sup_idx].squeeze().cpu().numpy()
                                        spt_img_2 = spt_img.copy()
                                        spt_img_2[spt_mask == 1] = colors[2]
                                        all_vis.append(spt_img_2)
                                    # for idx in slc_ids:
                                    for idx in range(sup_num,all_images.shape[0]):
                                        query_img = normalize_img(imgs_tensor_np[idx]) * 255  # 3,h,w
                                        query_img = query_img.transpose(1, 2, 0).astype(np.uint8)  # hwd
                                        img_vis.append(query_img)
                                        img_vis.append(query_img)
                                        vis_img2 = query_img.copy()
                                        vis_img3 = query_img.copy()
                                        pred = video_segments_bysup[idx][1]
                                        pred = pred.squeeze().cpu().numpy()
                                        vis_img2[pred > 0] = colors[0]
                                        label = mask_tensor[idx - sup_num].squeeze().cpu().numpy()
                                        vis_img3[label == 1] = colors[1]
                                        all_vis.append(vis_img2)
                                        all_vis.append(vis_img3)
                                    save_img = np.vstack((np.hstack(img_vis), np.hstack(all_vis)))
                                    cv2.imwrite(f'qry.jpg', save_img)
                                    assert 1 == 0
                                    imgs_tensor_np = imgs_tensor.cpu().numpy()  # d,3,h,w
                                    all_vis = []
                                    for idx in range(imgs_tensor.shape[0]):
                                        query_img = normalize_img(imgs_tensor_np[idx]) * 255  # 3,h,w
                                        query_img = query_img.transpose(1, 2, 0).astype(np.uint8)  # hw3
                                        vis_img1 = query_img.copy()
                                        vis_img2 = query_img.copy()
                                        vis_img3 = query_img.copy()
                                        vis_img4 = query_img.copy()
                                        pred = video_segments_bysup[idx + sup_num][1]
                                        pred = pred.squeeze().cpu().numpy()
                                        # vis_img1[pred > 0] = colors[0]
                                        if idx + sup_num in slc_ids:
                                            vis_img1[pred > 0] = colors[3]
                                        else:
                                            vis_img1[pred > 0] = colors[0]

                                        pred = video_segments[idx + sup_num][1]
                                        pred = pred.squeeze().cpu().numpy()
                                        vis_img2[pred > 0] = colors[1]

                                        pred = (1 - args.m_W) * video_segments[idx + sup_num][1] + args.m_W * \
                                               video_segments_bysup[idx + sup_num][
                                                   1]
                                        pred = pred.squeeze().cpu().numpy()
                                        vis_img3[pred > 0] = colors[4]

                                        msk = mask_tensor[idx].squeeze().cpu().numpy()
                                        vis_img4[msk == 1] = colors[2]

                                        all_vis.append(vis_img1)
                                        all_vis.append(vis_img2)
                                        all_vis.append(vis_img3)
                                        all_vis.append(vis_img4)

                                    num_cols = 8
                                    num_rows = (len(all_vis) + num_cols - 1) // num_cols  # 向上取整

                                    # 如果图像数量不能被5整除，用空白图像填充
                                    if len(all_vis) % num_cols != 0:
                                        padding_needed = num_cols - (len(all_vis) % num_cols)
                                        empty_img = np.zeros_like(all_vis[0])
                                        for _ in range(padding_needed):
                                            all_vis.append(empty_img)

                                    # 重新排列图像
                                    rows = []
                                    for i in range(num_rows):
                                        row = np.hstack(all_vis[i * num_cols: (i + 1) * num_cols])
                                        rows.append(row)
                                    save_img = np.vstack(rows)
                                    if 'sunseg' in args.dataset:
                                        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(f'qry1.jpg', save_img)
                                    # assert 1 == 0

                                # 生成query slice的prompt
                                query_bbox_dict = {}
                                query_msk_dict = {}
                                for slc_id in slc_ids:
                                    pred = video_segments_bysup[slc_id][1]
                                    pred = pred.squeeze().cpu().numpy()
                                    pred[pred > 0] = 1
                                    pred[pred < 0] = 0
                                    if np.max(pred) == 0:
                                        # TODO//
                                        print('no prediction result')
                                        continue
                                    else:
                                        bbox = generate_bbox(pred)
                                        query_msk_dict[slc_id] = pred
                                        query_bbox_dict[slc_id] = bbox
                                """begin inference"""
                                net.reset_state(eval_state)
                                for i in range(sup_num):
                                    _, out_obj_ids, out_mask_logits = net.add_new_mask(
                                        inference_state=eval_state,
                                        frame_idx=i,
                                        obj_id=1,
                                        mask=(sup_msks[i][0] == 1).int(),
                                    )
                                for slc_id in slc_ids:
                                    try:
                                        _, _, _ = net.add_new_mask(
                                            inference_state=eval_state,
                                            frame_idx=slc_id,
                                            obj_id=1,
                                            mask=query_msk_dict[slc_id],
                                        )
                                    except KeyError:
                                        print("adding box error, enter exception....")
                                        _, _, _ = net.add_new_mask(
                                            inference_state=eval_state,
                                            frame_idx=slc_id,
                                            obj_id=1,
                                            mask=torch.zeros(imgs_tensor.shape[2:]),
                                        )
                                video_segments = {}  # video_segments contains the per-frame segmentation results
                                video_segments_bysup = {}
                                ious_prop_dict = {}
                                ious_sup_dict = {}
                                for out_frame_idx, out_obj_ids, out_mask_logits, sup_out_mask_logits, obj_logit, obj_logit_sup, ious_prop, ious_sup in net.propagate_in_video(
                                        eval_state,
                                        start_frame_idx=start_slice):
                                    video_segments[out_frame_idx] = {
                                        out_obj_id: out_mask_logits[i]
                                        for i, out_obj_id in enumerate(out_obj_ids)
                                    }
                                    video_segments_bysup[out_frame_idx] = {
                                        out_obj_id: sup_out_mask_logits[i]
                                        for i, out_obj_id in enumerate(out_obj_ids)
                                    }
                                    ious_prop_dict[out_frame_idx] = ious_prop
                                    ious_sup_dict[out_frame_idx] = ious_sup
                                for out_frame_idx, out_obj_ids, out_mask_logits, sup_out_mask_logits, obj_logit, obj_logit_sup, ious_prop, ious_sup in net.propagate_in_video(
                                        eval_state,
                                        start_frame_idx=start_slice - 1,
                                        reverse=True):
                                    video_segments[out_frame_idx] = {
                                        out_obj_id: out_mask_logits[i]
                                        for i, out_obj_id in enumerate(out_obj_ids)
                                    }
                                    video_segments_bysup[out_frame_idx] = {
                                        out_obj_id: sup_out_mask_logits[i]
                                        for i, out_obj_id in enumerate(out_obj_ids)
                                    }
                                    ious_prop_dict[out_frame_idx] = ious_prop
                                    ious_sup_dict[out_frame_idx] = ious_sup
                            else:
                                start, end = obj_ranges[0], obj_ranges[1]
                                # 计算范围内的slice总数
                                total_slices = end - start + 1
                                # 取范围内slice总数和7中的较小值
                                num_slices = min(total_slices, 7)

                                prompt_slices = np.random.choice(np.arange(start, end + 1), size=num_slices, replace=False)
                                # 对选出的slice序号进行排序
                                prompt_slices = np.sort(prompt_slices)

                                if prompt == 'click':
                                    points = pt_dict[prompt_slice]
                                    labels = point_labels_dict[prompt_slice]
                                    _, _, _ = net.add_new_points(
                                        inference_state=eval_state,
                                        frame_idx=prompt_slice + sup_num,
                                        obj_id=1,
                                        points=points,
                                        labels=labels,
                                        clear_old_points=False,
                                    )
                                elif prompt == 'bbox':
                                    for prompt_slice in prompt_slices:
                                        try:
                                            bbox = bbox_dict[prompt_slice]
                                            _, _, _ = net.add_new_bbox(
                                                inference_state=eval_state,
                                                frame_idx=prompt_slice - obj_ranges[0] + sup_num,
                                                obj_id=1,
                                                bbox=bbox,
                                                clear_old_points=False,
                                            )
                                        except:
                                            pass

                                video_segments = {}  # video_segments contains the per-frame segmentation results
                                obj_logits_dict = {}
                                for out_frame_idx, out_obj_ids, out_mask_logits, sup_out_mask_logits, obj_logit, obj_logit_sup, _, _ in net.propagate_in_video(
                                        eval_state,
                                        start_frame_idx=prompt_slices[0]- obj_ranges[0] + sup_num):
                                    video_segments[out_frame_idx] = {
                                        out_obj_id: out_mask_logits[i]
                                        for i, out_obj_id in enumerate(out_obj_ids)
                                    }
                                    obj_logits_dict[out_frame_idx] = obj_logit
                                for out_frame_idx, out_obj_ids, out_mask_logits, sup_out_mask_logits, obj_logit, obj_logit_sup, _, _ in net.propagate_in_video(
                                        eval_state,
                                        start_frame_idx=prompt_slices[0]- obj_ranges[0] + sup_num - 1,
                                        reverse=True):
                                    video_segments[out_frame_idx] = {
                                        out_obj_id: out_mask_logits[i]
                                        for i, out_obj_id in enumerate(out_obj_ids)
                                    }
                                    obj_logits_dict[out_frame_idx] = obj_logit

                            """ evaluate """
                            prediction = np.full((imgs_tensor.shape[0], imgs_tensor.shape[-2], imgs_tensor.shape[-1]),
                                                 fill_value=0)

                            for id in range(sup_num, imgs_tensor.shape[0] + sup_num):
                                if args.test_prompt == 'bbox':
                                    pred = video_segments[id][1]
                                else:
                                    pred_bypro = video_segments[id][1]
                                    pred = pred_bypro
                                    pred_bysup = video_segments_bysup[id][1]
                                    # method0: by weighted average
                                    pred = args.m_W * pred_bysup + (1 - args.m_W) * pred_bypro
                                pred = pred.clone()
                                pred[pred > 0] = 1
                                pred[pred < 0] = 0
                                prediction[id - sup_num, :, :] = pred.squeeze().cpu().numpy()
                        else:
                            prediction = np.full((imgs_tensor.shape[0], imgs_tensor.shape[-2], imgs_tensor.shape[-1]),
                                                 fill_value=0)
                        # evaluate
                        label = mask_tensor.squeeze(1).cpu().numpy()
                        dice_cls = calculate_metric_percase(prediction == 1, label == 1)
                        print(f'case:{name}, dice of organ {organ_name}: {dice_cls}')
                        organ_dice[organ_name].append(dice_cls)
                        image = imgs_tensor.permute(1, 2, 3, 0)[0].cpu().numpy()  # h,w,d
                        label = label.transpose(1, 2, 0)  # h,w,d
                        prediction = prediction.transpose(1, 2, 0)  # h,w,d
                        if args.save_result:
                            imgs_tensor_np = imgs_tensor.cpu().numpy()  # d,3,h,w
                            all_vis = []
                            for idx in range(imgs_tensor.shape[0]):
                                query_img = normalize_img(imgs_tensor_np[idx]) * 255  # 3,h,w
                                query_img = query_img.transpose(1, 2, 0).astype(np.uint8)  # hwd
                                vis_img1 = query_img.copy()
                                vis_img2 = query_img.copy()
                                vis_img3 = query_img.copy()
                                vis_img4 = query_img.copy()
                                # bysupport
                                pred = video_segments_bysup[idx + sup_num][1]
                                pred = pred.squeeze().cpu().numpy()

                                if idx + sup_num in slc_ids:
                                    vis_img1[pred > 0] = colors[3]
                                else:
                                    vis_img1[pred > 0] = colors[0]
                                # byprop
                                pred = video_segments[idx + sup_num][1]
                                pred = pred.squeeze().cpu().numpy()
                                vis_img2[pred > 0] = colors[1]
                                # fuse
                                pred = prediction[..., idx]
                                vis_img3[pred > 0] = colors[4]
                                # gt
                                msk = label[..., idx]
                                vis_img4[msk == 1] = colors[2]
                                # 画框部分
                                # y_min, x_min, y_max, x_max = bbox_dict[idx - sup_num][1]
                                # cv2.rectangle(vis_img3, (y_min, x_min), (y_max, x_max), (0, 255, 0), 2)

                                all_vis.append(vis_img1)
                                all_vis.append(vis_img2)
                                all_vis.append(vis_img3)
                                all_vis.append(vis_img4)

                            # 每行5张图像
                            num_cols = 8
                            num_rows = (len(all_vis) + num_cols - 1) // num_cols  # 向上取整

                            # 如果图像数量不能被5整除，用空白图像填充
                            if len(all_vis) % num_cols != 0:
                                padding_needed = num_cols - (len(all_vis) % num_cols)
                                empty_img = np.zeros_like(all_vis[0])
                                for _ in range(padding_needed):
                                    all_vis.append(empty_img)
                            # 重新排列图像
                            rows = []
                            for i in range(num_rows):
                                row = np.hstack(all_vis[i * num_cols: (i + 1) * num_cols])
                                rows.append(row)

                            final_image = np.vstack(rows)
                            if 'sunseg' in args.dataset:
                                final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(f'qry_1.jpg', final_image)
                            assert 1 == 0

                            # print("save res!!!!!!!")
                            # nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                            #          os.path.join(test_save_path, name + "_pred({:.4f}).nii.gz".format(dice_cls)))
                            # nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)),
                            #          os.path.join(test_save_path, name + "_image.nii.gz"))
                            # nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)),
                            #          os.path.join(test_save_path, name + "_gt.nii.gz"))
                        pbar.update()

        organ_avg = defaultdict(list)
        organ_avg['meta info'] = [f'7 random'] #
        for key, value in organ_dice.items():
            if len(value) > 0:
                mean = np.mean(value)
                all_organ_avg.append(mean)
                std = np.std(value)
                all_organ_std.append(std)
                print(f'organ:{key}, dice: {mean:.3f}\u00B1{std:.3f}')
                newkey = f'{key}_{mean:.3f}_{std:.3f}'
                organ_avg[newkey] = value
                all_organ.append(f'{key}_{mean:.3f}_{std:.3f}')
            else:
                print(f'organ:{key}, dice: No values')
                newkey = f'{key}_NA'
                organ_avg[newkey] = []

        # 转换defaultdict为普通dict再写入
        output_dict = dict(organ_avg)
        with open(os.path.join(test_save_path, f'result_{organ}.json'), 'a', encoding='utf-8') as f:
            json.dump(output_dict, f, indent=4)
    res = {}
    res['meta info'] = [f'7 random']
    res['mean'] = np.mean(all_organ_avg)
    res['std'] = np.mean(all_organ_std)
    res['item'] = all_organ
    with open(os.path.join(test_save_path, f'result_all.json'), 'a', encoding='utf-8') as f:
        json.dump(res, f, indent=4)


if __name__ == '__main__':
    main()
