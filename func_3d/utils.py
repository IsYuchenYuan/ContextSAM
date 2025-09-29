"""Utility functions for training and evaluation.
    Yunli Qi
"""

import logging
import os
import random
import sys
import time
from datetime import datetime

import dateutil.tz
import numpy as np
import torch
from torch.autograd import Function

import cfg
from PIL import ImageEnhance, Image

args = cfg.parse_args()
device = torch.device('cuda', args.gpu_device)

dataset_organname = {
    'word': [
        "Liver",
        "Spleen",
        "Kidney L",
        "Kidney R",
        "Stomach",
        "Gallbladder",
        "Epsophagus",
        "Pancreas",
        "Duodenum",
        "Colon",
        "Intestine",
        "Adrenal R",
        "Adrenal L",
        "Rectum",
        "Bladder",
        "Head of Femur L",
        "Head of Femur R"
    ],
    'acdc': [
        "Left_Ventricle",
        "Myocardium",
        "Right_Ventricle"
    ],
    'T2SPIR': [
        "liver",
        "right_kidney",
        "left_kidney",
        "spleen"
    ],
    'T1DUAL_InPhase': [
        "liver",
        "right_kidney",
        "left_kidney",
        "spleen"
    ],
    'T1DUAL_OutPhase': [
        "liver",
        "right_kidney",
        "left_kidney",
        "spleen"
    ],
    'chaos-ct': [
        "liver",
    ],
    'ski10': [
        "Femur_Bone",
        "Femoral_Cartilage",
        "Tibia_Bone",
        "Tibial_Cartilage"
    ],
    'sunseg_hard': [
        "polyp_hard",
    ],
    'sunseg_easy': [
        "polyp_easy",
    ],
    'mmwhs': [
        "left ventricle",
        "right ventricle",
        "left atrium",
        "right atrium",
        "myocardium",
        "ascending aorta",
        "pulmonary artery"
    ],
    'camus': [
        "LV_endo",
        "LV_epi",
        "LA"
    ],
    'camus_2ch': [
        "LV_endo",
        "LV_epi",
        "LA"
    ],
    'camus_4ch': [
        "LV_endo",
        "LV_epi",
        "LA"
    ],
    'btcv': [
        "Spleen", "R.Kd", "L.Kd", "GB", "Eso", "Liver", "Stomach", "Aorta", "IVC", "Veins", "Pancreas", "R.AG", "L.AG"
    ],
    'hepaticvessel': [
        "vessel",
    ],
    'lungtumor':[
        "tumor",
    ]
}

ORGAN_LIST = ['01', '02', '03', '07', '08', '09', '10_07', '10_08', '10_09', '12', '13']
TEMPLATE = {
    '01': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    '02': [1, 3, 4, 5, 6, 7, 11, 14],
    '03': [6],
    '07': [6, 1, 3, 2, 7, 4, 5, 11, 14, 18, 19, 12, 13, 20, 21, 23, 24],
    '08': [6, 2, 3, 1, 11],
    '09': [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 21, 22],
    '12': [6, 21, 16, 17, 2, 3],
    '13': [6, 2, 3, 1, 11, 8, 9, 7, 4, 5, 12, 13, 25],
    '10_07': [11],
    '10_08': [15],
    '10_09': [1],
}
ORGAN_TO_DATASET = {1: ['01', '02', '07', '08', '09', '10_09', '13'], 2: ['01', '07', '08', '09', '12', '13'],
                    3: ['01', '02', '07', '08', '09', '12', '13'], 4: ['01', '02', '07', '09', '13'],
                    5: ['01', '02', '07', '09', '13'], 6: ['01', '02', '03', '07', '08', '09', '12', '13'],
                    7: ['01', '02', '07', '09', '13'], 8: ['01', '09', '13'], 9: ['01', '09', '13'], 10: ['01'],
                    11: ['01', '02', '07', '08', '09', '10_07', '13'], 12: ['01', '07', '09', '13'],
                    13: ['01', '07', '09', '13'], 14: ['01', '02', '07', '09'], 15: ['10_08'], 16: ['12'], 17: ['12'],
                    18: ['07'], 19: ['07'], 20: ['07'], 21: ['07', '09', '12'], 22: ['09'], 23: ['07'], 24: ['07'],
                    25: ['13']}

ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus',
              'Liver', 'Stomach', 'Aorta', 'Postcava', 'Portal Vein and Splenic Vein',
              'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
              'Right Lung', 'Left Lung', 'Colon', 'Intestine', 'Rectum',
              'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck', ]

DATASET_SIZE = {
    '01': 31,
    '02': 27,
    '03': 14,
    '07': 80,
    '08': 706,
    '09': 145,
    '10_07': 198,
    '10_08': 210,
    '10_09': 27,
    '12': 94,
    '13': 43
}

ORGAN_GROUPS = {
    0: [  # 高对比度、边界清晰的实质性器官
        'Liver',
        'Spleen',
        'Right Kidney',
        'Left Kidney',
        'Right Lung',
        'Left Lung'
    ],

    1: [  # 管状结构，呈条状、走行清晰
        'Aorta',
        'Postcava',
        'Portal Vein and Splenic Vein',
        'Hepatic Vessel',
        'Celiac Truck',
        'Esophagus'
    ],

    2: [  # 薄壁腔性器官，密度不均匀，可能含气体
        'Stomach',
        'Duodenum',
        'Colon',
        'Intestine',
        'Rectum',
        'Gall Bladder',
        'Bladder'
    ],

    3: [  # 小体积或边界模糊的器官
        'Pancreas',
        'Right Adrenal Gland',
        'Left Adrenal Gland',
        'Prostate',
        'Left Head of Femur',
        'Right Head of Femur'
    ]
}

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2
# from torchvision.transforms import ElasticTransform
import torchvision.transforms as transforms


def affine_transform(image, mask):
    """
    仿射变换：包括旋转、缩放、扭曲
    """
    h, w = image.shape[1:]

    # 随机旋转角度 (-30 到 30度)
    angle = np.random.uniform(-30, 30)

    # 随机缩放因子 (0.8 到 1.2)
    scale = np.random.uniform(0.8, 1.2)

    # 随机剪切因子
    shear = np.random.uniform(-0.3, 0.3)

    # 计算变换矩阵
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 添加剪切变换
    M[0, 1] += shear
    M[1, 0] += shear

    # 应用变换
    image_np = image.transpose(1, 2, 0)  # CHW -> HWC
    mask_np = mask.squeeze()

    transformed_image = cv2.warpAffine(image_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    transformed_mask = cv2.warpAffine(mask_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return torch.from_numpy(transformed_image.transpose(2, 0, 1)), torch.from_numpy(transformed_mask).unsqueeze(0)


def random_flip_rotate(img, msk):
    """对图像和掩码进行随机翻转（水平、垂直）和 90 度旋转"""
    if random.random() > 0.5:  # 50% 概率水平翻转
        img = np.flip(img, axis=2).copy()  # 假设通道顺序是 (C, H, W)
        msk = np.flip(msk, axis=2).copy()

    if random.random() > 0.5:  # 50% 概率垂直翻转
        img = np.flip(img, axis=1).copy()
        msk = np.flip(msk, axis=1).copy()

    # 随机 90 度旋转（0, 90, 180, 270 度）
    k = random.choice([0, 1, 2, 3])  # 选择旋转次数
    img = np.rot90(img, k, axes=(1, 2)).copy()  # 在 H-W 轴上旋转
    msk = np.rot90(msk, k, axes=(1, 2)).copy()  # 在 H-W 轴上旋转

    return img, msk


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def tensor_to_numpy(image_tensor):
    """将 PyTorch Tensor (C, H, W) 转换为 NumPy (H, W, C)"""
    image_numpy = image_tensor.numpy()  # 转换为 NumPy
    image_numpy = image_numpy.astype(np.uint8)  # 归一化到 0-255
    image_numpy = image_numpy.transpose(1, 2, 0)  # 变换为 (H, W, C)
    return image_numpy


def numpy_to_tensor(image_numpy):
    """将 NumPy (H, W, C) 转换回 PyTorch Tensor (C, H, W)"""
    image_tensor = torch.from_numpy(image_numpy.transpose(2, 0, 1))  # 变换回 (C, H, W)
    image_tensor = image_tensor.float()
    return image_tensor


def enhance_contrast_histogram_equalization(image_tensor):
    """使用直方图均衡化增强图像对比度"""
    image_numpy = tensor_to_numpy(image_tensor)  # 转换为 NumPy

    img_yuv = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2YUV)  # 转换到 YUV 颜色空间
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # 仅对亮度通道均衡化
    enhanced_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)  # 转换回 RGB

    return numpy_to_tensor(enhanced_image)  # 转换回 Tensor


def enhance_contrast_CLAHE(image_tensor):
    """使用 CLAHE 自适应直方图均衡化增强图像对比度"""
    image_numpy = tensor_to_numpy(image_tensor)  # 转换为 NumPy

    img_yuv = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2YUV)  # 转换到 YUV 颜色空间
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 创建 CLAHE 处理器
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])  # 仅对亮度通道增强
    enhanced_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)  # 转换回 RGB

    return numpy_to_tensor(enhanced_image)  # 转换回 Tensor


def augment_data(sup_imgs, sup_msks, aug):
    """
    Args:
        sup_imgs: shape (k,3,h,w) 的tensor，范围0-255
        sup_msks: shape (k,1,h,w) 的tensor
    Returns:
        aug_imgs: shape (k*3,3,h,w) 的tensor
        aug_msks: shape (k*3,1,h,w) 的tensor
    """

    k = sup_imgs.shape[0]
    aug_imgs = []
    aug_msks = []

    for i in range(k):
        img = sup_imgs[i]  # (3,h,w)
        msk = sup_msks[i]  # (1,h,w)

        # 原图
        aug_imgs.append(img)
        aug_msks.append(msk)

        if 'style' in aug:
            if 'rgb' in aug:
                # contrast = torch.rand(3, 1, 1) * 0.4 + 0.8  # 0.8-1.2
                # # 亮度范围从-50到+50改为-25到+25
                # brightness = (torch.rand(3, 1, 1) * 50 - 25)  # -25到+25
                # img_aug1 = img * contrast + brightness
                # img_aug1 = torch.clamp(img_aug1, 0, 255)
                # aug_imgs.append(img_aug1)
                # aug_msks.append(msk)
                enhanced_CLAHE_tensor = enhance_contrast_CLAHE(img)
                aug_imgs.append(enhanced_CLAHE_tensor)
                aug_msks.append(msk)
            else:
                # 对比度和亮度增强，伽马校正，适合不同vol风格有差异
                # contrast = torch.rand(1) * 1.7 + 0.3  # 0.3-2.0
                # brightness = torch.rand(1) * 200 - 100  # -100到+100
                # img_aug = img * contrast + brightness
                # gamma = torch.rand(1) * 1.5 + 0.5  # 0.5-2.0
                # img_aug = torch.pow(img_aug / 255.0, gamma) * 255  # 应用伽马校正
                # img_aug = torch.clamp(img_aug, 0, 255)  # 限制像素值在 0-255 之间
                contrast = torch.rand(1) * 0.8 + 0.6  # 0.6-1.4 (原本是0.3-2.0)
                brightness = torch.rand(1) * 100 - 50  # -50到+50 (原本是-100到+100)
                img_aug = img * contrast + brightness
                gamma = torch.rand(1) * 0.6 + 0.7  # 0.7-1.3 (原本是0.5-2.0)
                img_aug = torch.pow(img_aug / 255.0, gamma) * 255  # 应用伽马校正
                img_aug = torch.clamp(img_aug, 0, 255)  # 限制像素值在 0-255 之间
                aug_imgs.append(img_aug)
                aug_msks.append(msk)
        if 'shape' in aug:
            # 弹性形变 适合目标物体形状不固定
            img_np = img.numpy()
            msk_np = msk.numpy()
            img_elastic, msk_elastic = affine_transform(img_np, msk_np)
            aug_imgs.append(img_elastic)
            aug_msks.append(msk_elastic)
        if 'flip' in aug:
            # flip rotation 适合目标物体不固定，如tumor,polyp,lesion等
            img_np = img.numpy()
            msk_np = msk.numpy()
            img_flip_rot, msk_flip_rot = random_flip_rotate(img_np, msk_np)
            aug_imgs.append(torch.tensor(img_flip_rot))
            aug_msks.append(torch.tensor(msk_flip_rot))

    aug_imgs = torch.stack(aug_imgs)
    aug_msks = torch.stack(aug_msks)

    return aug_imgs.to(torch.uint8), aug_msks  # 确保返回uint8类型


def get_network(args, net, use_gpu=True, gpu_device=0, distribution=True):
    """ return given network
    """
    if net == 'sam2':
        from sam2.build_sam import build_sam2_video_predictor
        sam2_checkpoint = args.sam_ckpt
        model_cfg = args.sam_config
        net = build_sam2_video_predictor(config_file=model_cfg, ckpt_path=sam2_checkpoint, mode=None)
    elif net == 'sam2_finetune':
        from sam2_train.build_sam import build_sam2_video_predictor
        sam2_checkpoint = args.sam_ckpt
        model_cfg = args.sam_config
        net = build_sam2_video_predictor(config_file=model_cfg, ckpt_path=sam2_checkpoint, mode=None)
    elif net == 'sam2_xmem':
        from sam2_xmem_train.build_sam import build_sam2_video_predictor
        sam2_checkpoint = args.sam_ckpt
        model_cfg = args.sam_config
        net = build_sam2_video_predictor(config_file=model_cfg, ckpt_path=sam2_checkpoint, mode=None)
    elif net == 'sam2_xmem_spt':
        from sam2_xmem_spt_train.build_sam import build_sam2_video_predictor
        sam2_checkpoint = args.sam_ckpt
        model_cfg = args.sam_config
        net = build_sam2_video_predictor(config_file=model_cfg, ckpt_path=sam2_checkpoint, mode=None,
                                         lora_type=args.lora_type)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        net = net.to(device=gpu_device)

    return net


def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    # prefix = exp_path + '_' + timestamp
    prefix = exp_path
    os.makedirs(prefix, exist_ok=True)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path, exist_ok=True)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path, exist_ok=True)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path, exist_ok=True)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


def random_click(mask, point_labels=1, seed=None):
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    # max agreement position
    indices = np.argwhere(mask == max_label)
    # return point_labels, indices[np.random.randint(len(indices))]
    if seed is not None:
        rand_instance = random.Random(seed)
        rand_num = rand_instance.randint(0, len(indices) - 1)
    else:
        rand_num = random.randint(0, len(indices) - 1)
    output_index_1 = indices[rand_num][0]
    output_index_0 = indices[rand_num][1]
    return point_labels, np.array([output_index_0, output_index_1])


def generate_bbox(mask, variation=0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # check if all masks are black
    if len(mask.shape) != 2:
        current_shape = mask.shape
        raise ValueError(f"Mask shape is not 2D, but {current_shape}")
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    # max agreement position
    indices = np.argwhere(mask == max_label)
    # return point_labels, indices[np.random.randint(len(indices))]
    # print(indices)
    x0 = np.min(indices[:, 0])
    x1 = np.max(indices[:, 0])
    y0 = np.min(indices[:, 1])
    y1 = np.max(indices[:, 1])
    w = x1 - x0
    h = y1 - y0
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    if variation > 0:
        num_rand = [np.random.randn() * variation, np.random.randn() * variation]
        w *= 1 + num_rand[0]
        h *= 1 + num_rand[1]
        x1 = mid_x + w / 2
        x0 = mid_x - w / 2
        y1 = mid_y + h / 2
        y0 = mid_y - h / 2
    return np.array([y0, x0, y1, x1])


def eval_seg(pred, true_mask_p, threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0, 0, 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')
            cup_pred = vpred_cpu[:, 1, :, :].numpy().astype('int32')

            disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p[:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')

            '''iou for numpy'''
            iou_d += iou(disc_pred, disc_mask)
            iou_c += iou(cup_pred, cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()
            cup_dice += dice_coeff(vpred[:, 1, :, :], gt_vmask_p[:, 1, :, :]).item()

        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    elif c > 2:  # for multi-class segmentation > 2 classes
        ious = [0] * c
        dices = [0] * c
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            for i in range(0, c):
                pred = vpred_cpu[:, i, :, :].numpy().astype('int32')
                mask = gt_vmask_p[:, i, :, :].squeeze(1).cpu().numpy().astype('int32')

                '''iou for numpy'''
                ious[i] += iou(pred, mask)

                '''dice for torch'''
                dices[i] += dice_coeff(vpred[:, i, :, :], gt_vmask_p[:, i, :, :]).item()

        return tuple(np.array(ious + dices) / len(threshold))  # tuple has a total number of c * 2
    else:
        eiou, edice = 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')
            disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')

            '''iou for numpy'''
            eiou += iou(disc_pred, disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()

        return eiou / len(threshold), edice / len(threshold)


def iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device=input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


from medpy import metric


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum() == 0:
        return 0
    elif pred.sum() == 0 and gt.sum() > 0:
        return 0
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1


import numpy as np
from typing import List, Tuple


class RobustBoxMapper:
    def __init__(self,
                 position_var: float = 0.1,  # 位置变化范围（相对于图像大小的比例）
                 size_var: float = 0.2,  # 大小变化范围（相对于原始框大小的比例）
                 num_samples: int = 5):  # 采样框的数量
        self.position_var = position_var
        self.size_var = size_var
        self.num_samples = num_samples

    def get_robust_box(self,
                       template_box: List[float],
                       image_size: Tuple[int, int]) -> List[float]:
        """
        生成一个考虑位置和大小变化的框

        Args:
            template_box: 原始框 [x1, y1, x2, y2]
            image_size: 图像尺寸 (height, width)

        Returns:
            expanded_box: 扩展后的框 [x1, y1, x2, y2]
        """
        h, w = image_size
        x1, y1, x2, y2 = template_box

        # 计算原始框的中心点和大小
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        # 位置扰动范围
        pos_var_x = w * self.position_var
        pos_var_y = h * self.position_var

        # 大小扰动范围
        size_var_w = width * self.size_var
        size_var_h = height * self.size_var

        # 采样多个框
        boxes = []
        for _ in range(self.num_samples):
            # 添加位置扰动
            new_center_x = center_x + np.random.uniform(-pos_var_x, pos_var_x)
            new_center_y = center_y + np.random.uniform(-pos_var_y, pos_var_y)

            # 添加大小扰动
            new_width = width + np.random.uniform(-size_var_w, size_var_w)
            new_height = height + np.random.uniform(-size_var_h, size_var_h)

            # 计算新框
            new_x1 = new_center_x - new_width / 2
            new_y1 = new_center_y - new_height / 2
            new_x2 = new_center_x + new_width / 2
            new_y2 = new_center_y + new_height / 2

            boxes.append([new_x1, new_y1, new_x2, new_y2])

        # 计算包含所有框的最小外接框
        boxes = np.array(boxes)
        x1_min = np.min(boxes[:, 0])
        y1_min = np.min(boxes[:, 1])
        x2_max = np.max(boxes[:, 2])
        y2_max = np.max(boxes[:, 3])

        # 确保框在图像范围内
        x1_min = max(0, min(x1_min, w))
        y1_min = max(0, min(y1_min, h))
        x2_max = max(0, min(x2_max, w))
        y2_max = max(0, min(y2_max, h))

        return [int(x1_min), int(y1_min), int(x2_max), int(y2_max)]
