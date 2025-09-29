import torch
from torch.utils.data import Dataset, ConcatDataset
import os
from PIL import Image
from func_3d.utils import random_click, generate_bbox, TEMPLATE, ORGAN_LIST, ORGAN_NAME, DATASET_SIZE, ORGAN_TO_DATASET
import cv2
import nibabel as nib
import numpy as np
from PIL import ImageEnhance
import random


def remove_discontinuous(z_index):
    if len(z_index) <= 1:
        return z_index

    result = []
    for i in range(len(z_index)):
        is_continuous = False

        # Check with previous number
        if i > 0 and abs(z_index[i] - z_index[i - 1]) == 1:
            is_continuous = True

        # Check with next number
        if i < len(z_index) - 1 and abs(z_index[i] - z_index[i + 1]) == 1:
            is_continuous = True

        if is_continuous:
            result.append(z_index[i])

    return result

class DatasetTest(Dataset):
    def __init__(self, args, dataset, data_path, organ, support_num, sup_vol_num=3,transform=None, mode='Test',
                 prompt='click',
                 seed=None,
                 variation=0):

        dataset_modality = {
            'btcv': 'ct', 'word': 'ct', 'HaN': 'ct', 'mmwhs': 'ct', 'chaos-ct': 'ct', 'lungtumor':'ct',
            'T2SPIR': 'mri', 'T1DUAL_InPhase': 'mri', 'T1DUAL_OutPhase': 'mri', 'acdc': 'mri', 'ski10': 'mri',
            'camus': 'us', 'camus_2ch': 'us', 'camus_4ch': 'us', 'sunseg': 'colonscope'}

        if dataset_modality[dataset] == 'ct':
            print('The dataset modality is CT, using predefined window level')
            self.lower_bound = -175
            self.upper_bound = 250
        elif dataset_modality[dataset] == 'mri':
            print('The dataset modality is MRI, using percentile clip')
            self.lower_bound = self.upper_bound = 'percentile_clip'
        else:
            print('The dataset modality is not MRI or CT, no need clip')
            self.lower_bound = self.upper_bound = None

        qry_list = f'{data_path}/query_img_list.txt'
        with open(qry_list, 'r') as f:
            self.qry_list = [line.strip() for line in f.readlines()]

        qry_label_list = f'{data_path}/query_lb_list.txt'
        with open(qry_label_list, 'r') as f:
            self.qry_label_list = [line.strip() for line in f.readlines()]

        # Set the basic information of the dataset
        self.data_path = data_path
        self.organ = organ
        self.support_num = support_num
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.seed = seed
        self.variation = variation
        self.mode = mode
        self.dataset = dataset
        self.args = args

        # get support pairs
        self.sup_set = self._get_sup_paris()

    def _get_sup_paris(self):
        newsize = (self.img_size, self.img_size)

        """Get the support images"""
        if self.dataset in ['T2SPIR', 'T1DUAL_InPhase', 'T1DUAL_OutPhase']:
            tmp_img_paths = [tmp_path.replace('T1DUAL_InPhase', self.dataset) for tmp_path in self.tmp_list]
            tmp_mask_paths = [path.replace('mri_', 'segmentation_') for path in tmp_img_paths]
        else:
            tmp_img_paths = self.tmp_list
            tmp_mask_paths = self.tmp_label_list

        self.tmp_names = [name.split('/')[-1] for name in tmp_img_paths]

        sup_imgs = []
        sup_msks = []
        mid_stack_indices = []  # 存储stack后的中间slice索引
        cumulative_count = 0  # 记录已添加的slice数量

        # 统计3个vol的slice的总数
        slice_counter = 0
        index_dict = {}
        for i, tmp_img_path in enumerate(tmp_img_paths):
            vol = nib.as_closest_canonical(nib.load(tmp_img_path)).get_fdata()  # h,w,d
            if self.lower_bound == 'percentile_clip':
                lower_bound = np.percentile(vol, 0.5)
                upper_bound = np.percentile(vol, 99.5)
                # 使用这些分位数进行裁剪
                image_data_pre = np.clip(vol, lower_bound, upper_bound).astype(np.float32)
            elif self.lower_bound is None:
                image_data_pre = vol
            else:
                image_data_pre = np.clip(vol, self.lower_bound, self.upper_bound).astype(np.float32)  # 明确数据类型
            # 计算一次min和max，避免重复计算
            min_val = np.min(image_data_pre)
            max_val = np.max(image_data_pre)
            data_3d = ((image_data_pre - min_val) / (max_val - min_val) * 255.0)
            gt = nib.as_closest_canonical(nib.load(tmp_mask_paths[i])).get_fdata()
            slice_counter += gt.shape[-1]
            gt[gt != int(self.organ)] = 0
            gt[gt == int(self.organ)] = 1


            gt[gt > 0] = 1
            _, _, z_index = np.where(gt == 1)
            z_index = np.unique(z_index)
            z_index = remove_discontinuous(z_index)
            if len(z_index) == 0:
                print('this support volume has no target object')
                continue

            starting_frame_nonzero = z_index[0]
            end_frame_nonzero = z_index[-1]
            hasobj_num_frame = end_frame_nonzero - starting_frame_nonzero + 1
            tmp_indexes = list(range(starting_frame_nonzero, end_frame_nonzero + 1))
            # 计算中间slice在stack后的索引
            mid_frame = starting_frame_nonzero + hasobj_num_frame // 2
            mid_offset = tmp_indexes.index(mid_frame)  # 中间frame在tmp_indexes中的位置
            mid_stack_indices.append(cumulative_count + mid_offset)  # 加上之前累积的slice数量
            # import ipdb
            # ipdb.set_trace()
            for tmp_index in tmp_indexes:
                sup_img = data_3d[..., tmp_index]
                sup_img = np.stack([sup_img, sup_img, sup_img], axis=0)  # 3,h,w
                sup_img = cv2.resize(sup_img.transpose(1, 2, 0), newsize,
                                     interpolation=cv2.INTER_LINEAR)  # 转置为 (h, w, c)
                sup_img = torch.from_numpy(sup_img).permute(2, 0, 1)  # 转回 (c, h, w)
                sup_mask = gt[:, :, tmp_index]  # h,w
                sup_mask = cv2.resize(sup_mask, newsize, interpolation=cv2.INTER_NEAREST)
                sup_mask = torch.from_numpy(sup_mask).unsqueeze(0)  # 1,h,w
                sup_imgs.append(sup_img)
                sup_msks.append(sup_mask)
            cumulative_count += len(tmp_indexes)  # 更新累积的slice数量
            index_dict[tmp_img_path] = [starting_frame_nonzero, cumulative_count]

        # print(f'dataset:{self.dataset},slice number:{slice_counter}')
        sup_imgs = torch.stack(sup_imgs, dim=0)  # (k, c, h, w)
        sup_msks = torch.stack(sup_msks, dim=0)  # (k, 1, h, w)

        return sup_imgs, sup_msks, mid_stack_indices, index_dict

    def __len__(self):
        return len(self.qry_list)

    def __getitem__(self, index):

        newsize = (self.img_size, self.img_size)
        """Get the query images"""
        if self.dataset in ['T2SPIR', 'T1DUAL_InPhase', 'T1DUAL_OutPhase']:
            img_path = self.qry_list[index]
            img_path = img_path.replace('T1DUAL_InPhase', self.dataset)
            mask_path = img_path.replace('mri_', 'segmentation_')
            name = img_path.split('/')[-1].split('.')[0]
        else:
            img_path = self.qry_list[index]
            mask_path = self.qry_label_list[index]
            name = img_path.split('/')[-1].split('.')[0]
        vol = nib.as_closest_canonical(nib.load(img_path)).get_fdata()  # h,w,d
        num_frames = vol.shape[-1]
        # clip intensity
        if self.lower_bound == 'percentile_clip':
            lower_bound = np.percentile(vol, 0.5)
            upper_bound = np.percentile(vol, 99.5)
            # 使用这些分位数进行裁剪
            image_data_pre = np.clip(vol, lower_bound, upper_bound).astype(np.float32)
        elif self.lower_bound is None:
            image_data_pre = vol
        else:
            image_data_pre = np.clip(vol, self.lower_bound, self.upper_bound).astype(np.float32)  # 明确数据类型
        # 计算一次min和max，避免重复计算
        min_val = np.min(image_data_pre)
        max_val = np.max(image_data_pre)
        data_3d = ((image_data_pre - min_val) / (max_val - min_val) * 255.0)

        gt = nib.as_closest_canonical(nib.load(mask_path)).get_fdata()
        gt[gt != int(self.organ)] = 0
        gt[gt == int(self.organ)] = 1
        gt[gt > 0] = 1
        _, _, z_index = np.where(gt == 1)
        z_index = remove_discontinuous(z_index)
        if len(z_index) == 0:
            prompt_id = -1
            obj_ranges = [-1, -1]
        else:
            z_index = np.unique(z_index)
            starting_frame_nonzero = z_index[0]
            end_frame_nonzero = z_index[-1]
            hasobj_num_frame = end_frame_nonzero - starting_frame_nonzero + 1
            prompt_id = starting_frame_nonzero + hasobj_num_frame // 2
            obj_ranges = [starting_frame_nonzero, end_frame_nonzero]

        img_arr = np.zeros((*newsize, num_frames))  # h,w,d
        label_arr = np.zeros((*newsize, num_frames))  # h,w,d
        for i in range(num_frames):
            img = data_3d[:, :, i]  # data_3d:[1, h,w,d]
            img = cv2.resize(img, newsize, interpolation=cv2.INTER_LINEAR)
            assert len(np.array(img).shape) == 2
            mask = gt[..., i]
            mask = cv2.resize(mask, newsize, interpolation=cv2.INTER_NEAREST)
            img_arr[:, :, i] = np.float32(img)  # h,w
            label_arr[:, :, i] = np.float32(mask)

        if len(z_index) != 0:
            # 返回最大外接框
            rows = np.any(label_arr, axis=(1, 2))  # 在y,z维度投影到x轴
            cols = np.any(label_arr, axis=(0, 2))  # 在x,z维度投影到y轴

            # 获取极值坐标
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            hw_max_bbox = [x_min, y_min, x_max, y_max]
        else:
            hw_max_bbox = [0, 0, 0, 0]

        img_tensor = torch.from_numpy(img_arr.astype(np.float32))
        img_tensor = img_tensor.permute(2, 0, 1)  # d,h,w
        img_tensor = img_tensor.unsqueeze(1).repeat(1, 3, 1, 1)  # d,3,h,w
        label = torch.from_numpy(label_arr).long()
        label_tensor = label.permute(2, 0, 1)  # d,h,w
        # import ipdb
        # ipdb.set_trace()

        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}  # {'frame_idx',coordinates}

        for i in range(num_frames):
            mask = label_tensor[i].numpy()
            if self.prompt == 'click':
                point_label_dict[i], pt_dict[i] = random_click(mask, 1, seed=None)
            if self.prompt == 'bbox':
                bbox_dict[i] = generate_bbox(mask, variation=self.variation,
                                             seed=self.seed)

        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': label_tensor,
                'sup_imgs': self.sup_set[0],
                'sup_msks': self.sup_set[1],
                'sup_mid_stack_indices': self.sup_set[2],
                'image_meta_dict': image_meta_dict,
                'bbox': bbox_dict,
                'prompt_id_inframe': prompt_id,
                'obj_ranges': np.array(obj_ranges),
                'hw_max_bbox': np.array(hw_max_bbox)

            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': label_tensor,
                'sup_imgs': self.sup_set[0],
                'sup_msks': self.sup_set[1],
                'sup_mid_stack_indices': self.sup_set[2],
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
                'prompt_id_inframe': prompt_id,
                'obj_ranges': np.array(obj_ranges)
            }
        elif self.prompt == 'None' or self.prompt == 'mask':
            return {
                'image': img_tensor,
                'label': label_tensor,
                'sup_imgs': self.sup_set[0],
                'sup_msks': self.sup_set[1],
                'sup_mid_stack_indices': self.sup_set[2],
                'image_meta_dict': image_meta_dict,
                'prompt_id_inframe': prompt_id,
                'obj_ranges': np.array(obj_ranges),
                'tmp_names': self.tmp_names,
                'index_dict': self.sup_set[3]
            }
