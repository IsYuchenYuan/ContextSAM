# train.py
# !/usr/bin/env	python3


import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import numpy as np
import cv2

import cfg
from sam2_matcher.matcher.Matcher import build_matcher_oss
from func_3d.utils import eval_seg

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

matcher = build_matcher_oss(args)

datatransform = transforms.Compose([
    transforms.Resize(size=(args.matcher_img_size, args.matcher_img_size)),
    transforms.ToTensor()
])


def extract_bounding_box(image, mask=None):
    # 读取图像并转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 将图像转换为 8 位
    image_8bit = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    image_8bit = np.uint8(image_8bit)

    # Otsu 阈值分割
    _, thresh = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=2)

    # 找到轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 按面积排序并选择最大的几个轮廓，有些图片前景不连通
    # 计算所有轮廓的面积
    areas = [cv2.contourArea(c) for c in contours]
    # 找到最大面积
    max_area = max(areas)
    # 保留与最大面积相差不大的轮廓
    filtered_contours = [c for c, a in zip(contours, areas) if a > 0.7 * max_area]

    # 找到最大轮廓
    # max_contour = max(contours, key=cv2.contourArea)
    # 合并轮廓点
    all_points = np.concatenate(filtered_contours)
    # 计算外接矩形
    x, y, w, h = cv2.boundingRect(all_points)
    # 裁剪矩形区域
    cropped = image[y:y + h, x:x + w]
    if mask is not None:
        mask = mask[y:y + h, x:x + w]
        return cropped, mask
    return cropped


def detect_object(template_image, candidate_image, template_mask, grid_count=(128, 128)):
    template_h, template_w = template_image.shape[:2]
    grid_h, grid_w = template_h / grid_count[0], template_w / grid_count[1]
    qry_grid_h, qry_grid_w = candidate_image.shape[0] / grid_count[0], candidate_image.shape[1] / grid_count[1]
    # 找到模板中物体所在的最小网格
    template_indices = np.argwhere(template_mask)
    min_y, min_x = np.min(template_indices, axis=0) // [grid_h, grid_w]
    max_y, max_x = np.max(template_indices, axis=0) // [grid_h, grid_w]
    query_box = int(min_x * qry_grid_w), int(min_y * qry_grid_h), int((max_x - min_x + 1) * qry_grid_w), int(
        (max_y - min_y + 1) * qry_grid_h)
    return query_box


def crop_image(spt_path, qry_path, spt_msk_path, cls_num):
    support_msk = np.load(spt_msk_path)
    support_msk[support_msk != cls_num] = 0
    support_msk[support_msk == cls_num] = 1  # h,w
    template_img = cv2.imread(spt_path)
    template_image_crop, template_mask_crop = extract_bounding_box(template_img, support_msk)
    candidate_image = cv2.imread(qry_path)
    candidate_image_crop = extract_bounding_box(candidate_image)
    return template_image_crop, template_mask_crop, candidate_image_crop


def calculate_point_ratio(points, bbox):
    # 定义目标框 (x, y, width, height)
    x, y, w, h = bbox

    # 计算框内的点数
    points_in_box = 0
    for point in points:
        px, py = point
        if x <= px < x + w and y <= py < y + h:
            points_in_box += 1

    # 总点数
    total_points = len(points)

    # 计算比例
    if total_points == 0:
        return 0
    ratio = points_in_box / total_points
    return ratio


def measure_dbscan_clustering(points, bbox, eps=14, min_samples=3):
    # 提取框内的点
    x, y, w, h = bbox
    points_in_box = [(px, py) for (px, py) in points if x <= px < x + w and y <= py < y + h]

    # 转换为NumPy数组
    points_array = np.array(points_in_box)

    if len(points_array) < min_samples:
        return 0

    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)

    # 获取聚类标签
    labels = dbscan.labels_

    # 计算噪声点比例
    noise_ratio = np.sum(labels == -1) / len(labels)

    return 1 - noise_ratio


def cvtPIL(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image.astype(np.uint8))
    return image


def is_point_inside_bbox(point, bbox):
    x, y, w, h = bbox
    px, py = point
    return x <= px < x + w and y <= py < y + h


def expand_bbox(bbox, points):
    x, y, w, h = bbox
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    new_x = min(x, min_x)
    new_y = min(y, min_y)
    new_w = max(x + w, max_x) - new_x
    new_h = max(y + h, max_y) - new_y
    return (new_x, new_y, new_w, new_h)


def point_to_bbox_distance(point, bbox):
    x, y, w, h = bbox
    px, py = point

    # 点在框内
    if x <= px <= x + w and y <= py <= y + h:
        return 0

    # 点在框左右侧
    if x <= px <= x + w:
        return min(abs(py - y), abs(py - (y + h)))

    # 点在框上下侧
    if y <= py <= y + h:
        return min(abs(px - x), abs(px - (x + w)))

    # 点在框四个角外
    if px < x and py < y:
        return np.linalg.norm((px - x, py - y))
    if px > x + w and py < y:
        return np.linalg.norm((px - (x + w), py - y))
    if px < x and py > y + h:
        return np.linalg.norm((px - x, py - (y + h)))
    if px > x + w and py > y + h:
        return np.linalg.norm((px - (x + w), py - (y + h)))


def expand_box_to_cover(points, bbox, threshold_distance=20, min_cluster_points=5, eps=14, min_samples=3):
    # 使用DBSCAN进行聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    # print(set(labels))
    x, y, w, h = bbox
    expanded_bbox = bbox

    for label in set(labels):
        if label == -1:
            continue
        cluster_points = points[labels == label]
        # 检查聚类是否部分在框内和框外
        inside = any(is_point_inside_bbox(p, expanded_bbox) for p in cluster_points)
        outside = any(not is_point_inside_bbox(p, expanded_bbox) for p in cluster_points)
        # print(f'label{label}    inside:{inside} outside:{outside} num:{len(cluster_points)}')
        if inside and outside:
            # 扩展方框
            expanded_bbox = expand_bbox(expanded_bbox, cluster_points)

        # 检查框外聚类并满足条件
        if not inside and len(cluster_points) >= min_cluster_points:
            distances = [point_to_bbox_distance(p, expanded_bbox) for p in cluster_points]
            if any(d < threshold_distance for d in distances):
                expanded_bbox = expand_bbox(expanded_bbox, cluster_points)
            # print(distances)
    return expanded_bbox


def matcher_selection(spt_path, spt_msk_path, qry_vol_path, cls_num):
    GPUdevice = args.gpu_device
    sim_vol = []
    num_frame = len(os.listdir(qry_vol_path))
    for i in range(num_frame):
        # 1. 裁去黑边
        qry_img_path = os.path.join(qry_vol_path, f'{i}.jpg')
        template_image_crop, template_mask_crop, candidate_image_crop = crop_image(spt_path, qry_img_path, spt_msk_path,
                                                                                   cls_num)
        # 2. 根据support图片中的目标物体生成box
        query_box = detect_object(template_image_crop, candidate_image_crop, template_mask_crop)
        original_height, original_width = candidate_image_crop.shape[:2]
        new_height, new_width = args.matcher_img_size, args.matcher_img_size
        # 计算缩放比例
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        # 调整目标框
        x, y, w, h = query_box
        new_x = int(x * scale_x)
        new_y = int(y * scale_y)
        new_w = int(w * scale_x)
        new_h = int(h * scale_y)
        scaled_bbox = (new_x, new_y, new_w, new_h)

        # 3. Matcher prepare references and target
        support_imgs = datatransform(cvtPIL(template_image_crop)).unsqueeze(0).unsqueeze(0)  # b,n,3,h,w
        support_masks = torch.from_numpy(template_mask_crop).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # b,n,1,h,w
        support_masks = F.interpolate(support_masks.squeeze(2).float(), support_imgs.size()[-2:],
                                      mode='nearest').unsqueeze(2)
        query_img = datatransform(cvtPIL(candidate_image_crop)).unsqueeze(0)  # b,3,h,w
        query_img = query_img.to(GPUdevice)
        support_imgs = support_imgs.to(GPUdevice)
        support_masks = support_masks.to(GPUdevice)

        matcher.set_reference(support_imgs, support_masks)  # 1,n,c,h,w
        matcher.set_target(query_img)

        # 2. Predict mask of target
        points, box, C = matcher.matching()  # (rangenumber,max_iterations, i, 2)
        if len(points) < 5:
            similarity = -1
            sim_vol.append(similarity)
            continue
        scaled_bbox = expand_box_to_cover(points, scaled_bbox)
        inbox_ratio = calculate_point_ratio(points, scaled_bbox)
        cluster_degree = measure_dbscan_clustering(points, scaled_bbox)
        similarity = inbox_ratio + 0.3 * cluster_degree
        # print(f'slice{i}: similarity:{similarity}   inbox_ratio:{inbox_ratio}   cluster:{cluster_degree}')
        sim_vol.append(similarity)
    return np.argmax(sim_vol), np.argsort(sim_vol)[-5:]


def matcher_generate(spt_path, spt_msk_path, qry_vol_path, topk, cls_num):
    GPUdevice = args.gpu_device
    support_img = Image.open(spt_path).convert('RGB')
    support_imgs = datatransform(support_img).unsqueeze(0).unsqueeze(0)  # b,n,3,h,w
    support_msk = np.load(spt_msk_path)
    support_msk[support_msk != cls_num] = 0
    support_msk[support_msk == cls_num] = 1
    support_masks = torch.from_numpy(support_msk).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # b,n,1,h,w
    support_masks = F.interpolate(support_masks.squeeze(2).float(), support_imgs.size()[-2:],
                                  mode='nearest').unsqueeze(2)
    support_imgs = support_imgs.to(GPUdevice)
    support_masks = support_masks.to(GPUdevice)

    res = {}
    mask = {}
    for i in topk[::-1]:
        qry_img_path = os.path.join(qry_vol_path, str(i) + ".jpg")
        qry_msk_path = os.path.join(qry_vol_path.replace('image', 'mask'), str(i) + ".npy")
        query_img = Image.open(qry_img_path).convert('RGB')
        h, w = np.array(query_img).shape[:2]
        query_msk = np.load(qry_msk_path)
        query_msk[query_msk != cls_num] = 0
        query_msk[query_msk == cls_num] = 1
        query_img = datatransform(query_img).unsqueeze(0)  # b,3,h,w
        query_mask = torch.from_numpy(query_msk).unsqueeze(0).unsqueeze(0)  # b,1,h,w
        query_mask = F.interpolate(query_mask.float(), query_img.size()[-2:],
                                   mode='nearest').squeeze(1)  # b,h,w
        query_img = query_img.to(GPUdevice)
        query_mask = query_mask.to(GPUdevice)
        # 1. Matcher prepare references and target
        matcher.set_reference(support_imgs, support_masks)  # 1,n,c,h,w
        matcher.set_target(query_img)

        # 2. Predict mask of target
        all_points, box, C = matcher.matching()
        pred_msk, _ = matcher.predict(all_points, C)
        pred_msk = pred_msk.unsqueeze(0)
        pred = (pred_msk > 0).float()  # 1,1,H,W
        ori_scale_pred = F.interpolate(pred, (h, w), mode='nearest')
        mask[i] = ori_scale_pred.squeeze()  # H,W
        pred = torch.sigmoid(pred)
        temp = eval_seg(pred, query_mask.unsqueeze(0), [0.5])
        res[i] = temp[1]
        matcher.clear()
        if temp[1] > 0.8:
            break
    p_slice = max(res, key=res.get)
    p_mask = mask[p_slice] # h,w [0-1]
    return p_slice, p_mask, res[p_slice]
