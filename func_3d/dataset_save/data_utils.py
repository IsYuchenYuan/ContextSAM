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
            'btcv': 'ct', 'word': 'ct', 'HaN': 'ct', 'mmwhs': 'ct', 'chaos-ct': 'ct',
            'T2SPIR': 'mri', 'T1DUAL_InPhase': 'mri', 'T1DUAL_OutPhase': 'mri', 'acdc': 'mri', 'ski10': 'mri',
            'camus': 'us', 'camus_2ch': 'us', 'camus_4ch': 'us', 'sunseg': 'colonscope'}

        if dataset_modality[dataset] == 'ct':
            print('The dataset modality is CT, using predefined window level')
            self.lower_bound = -175
            self.upper_bound = 250
            # self.lower_bound = -200
            # self.upper_bound = 800
        elif dataset_modality[dataset] == 'mri':
            print('The dataset modality is MRI, using percentile clip')
            self.lower_bound = self.upper_bound = 'percentile_clip'
        else:
            print('The dataset modality is not MRI or CT, no need clip')
            self.lower_bound = self.upper_bound = None

        # qry_list = f'{data_path}/query_img_list_{sup_vol_num}.txt'
        qry_list = f'{data_path}/query_img_list.txt'
        with open(qry_list, 'r') as f:
            self.qry_list = [line.strip() for line in f.readlines()]
            # self.qry_list = ['/research/d5/gds/ycyuan22/code/Medical-SAM2-main/data/Train_organ/01_Multi-Atlas_Labeling/img/img0004.nii.gz']
            # self.qry_list = ['/research/d5/gds/ycyuan22/code/Medical-SAM2-main/data/Train_organ/ACDC/database/training/patient005/patient005_frame01.nii/DCM10Gate1.nii']
        # qry_label_list = f'{data_path}/query_lb_list_{sup_vol_num}.txt'
        qry_label_list = f'{data_path}/query_lb_list.txt'
        with open(qry_label_list, 'r') as f:
            self.qry_label_list = [line.strip() for line in f.readlines()]
            # self.qry_label_list = ['/research/d5/gds/ycyuan22/code/Medical-SAM2-main/data/Train_organ/01_Multi-Atlas_Labeling/label/label0004.nii.gz']
            # self.qry_label_list = ['/research/d5/gds/ycyuan22/code/Medical-SAM2-main/data/Train_organ/ACDC/database/training/patient005/patient005_frame01_gt.nii/DCM10-OH-AL_V2_1.nii']
        # tmp_list = f'{data_path}/support_img_list_{sup_vol_num}.txt'
        tmp_list = f'{data_path}/support_img_list.txt'
        with open(tmp_list, 'r') as f:
            self.tmp_list = [line.strip() for line in f.readlines()]
        # tmp_label_list = f'{data_path}/support_lb_list_{sup_vol_num}.txt'
        tmp_label_list = f'{data_path}/support_lb_list.txt'
        with open(tmp_label_list, 'r') as f:
            self.tmp_label_list = [line.strip() for line in f.readlines()]


        # This is for cross-domain referring word-btcv
        # tmp_list = f'./data/Test_organ/word/support_img_list.txt'
        # with open(tmp_list, 'r') as f:
        #     self.tmp_list = [line.strip() for line in f.readlines()]
        # tmp_label_list = f'./data/Test_organ/word/support_lb_list.txt'
        # with open(tmp_label_list, 'r') as f:
        #     self.tmp_label_list = [line.strip() for line in f.readlines()]
        # self.organ_map = {1: 2, 2: 4, 3: 3, 4: 6, 5: 7, 6: 1, 7: 5, 11: 8, 12: 12, 13: 13}

        # This is for cross-domain referring btcv-word
        # tmp_list = f'./data/Test_organ/btcv/support_img_list.txt'
        # with open(tmp_list, 'r') as f:
        #     self.tmp_list = [line.strip() for line in f.readlines()]
        # tmp_label_list = f'./data/Test_organ/btcv/support_lb_list.txt'
        # with open(tmp_label_list, 'r') as f:
        #     self.tmp_label_list = [line.strip() for line in f.readlines()]
        # self.organ_map = {2: 1, 4: 2, 3: 3, 6: 4, 7: 5, 1: 6, 5: 7, 8: 11, 12: 12, 13: 13}

        # # This is for cross-domain referring t1in-ct
        # tmp_list = f'./data/Test_organ/CHAOS_MRI/support_img_list.txt'
        # with open(tmp_list, 'r') as f:
        #     self.tmp_list = [line.strip() for line in f.readlines()]
        # tmp_label_list = f'./data/Test_organ/CHAOS_MRI/support_lb_list.txt'
        # with open(tmp_label_list, 'r') as f:
        #     self.tmp_label_list = [line.strip() for line in f.readlines()]

        # This is for cross-domain referring ct-t1in
        # tmp_list = f'./data/Test_organ/chaos-ct/support_img_list.txt'
        # with open(tmp_list, 'r') as f:
        #     self.tmp_list = [line.strip() for line in f.readlines()]
        # tmp_label_list = f'./data/Test_organ/chaos-ct/support_lb_list.txt'
        # with open(tmp_label_list, 'r') as f:
        #     self.tmp_label_list = [line.strip() for line in f.readlines()]


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
            # for cross-domain referring
            # tmp_img_paths = [tmp_path.replace('T1DUAL_InPhase', 'T2SPIR') for tmp_path in self.tmp_list]
            # tmp_mask_paths = [path.replace('mri_', 'segmentation_') for path in tmp_img_paths]
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

class DatasetTestOneshot(Dataset):
    def __init__(self, args, dataset, data_path, organ, support_num, sup_vol_num=3,transform=None, mode='Test',
                 prompt='click',
                 seed=None,
                 variation=0):

        dataset_modality = {
            'btcv': 'ct', 'word': 'ct', 'HaN': 'ct', 'mmwhs': 'ct', 'chaos-ct': 'ct',
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

        if args.fold == 0:
            self.qry_list = ['./data/CHAOST2/chaos_MR_T2_normalized/image_1.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_2.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_5.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_8.nii.gz',]
            self.qry_label_list = ['./data/CHAOST2/chaos_MR_T2_normalized/label_1.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/label_2.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/label_5.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/label_8.nii.gz', ]
            self.tmp_list = ['./data/CHAOST2/chaos_MR_T2_normalized/image_3.nii.gz']
            self.tmp_label_list = ['./data/CHAOST2/chaos_MR_T2_normalized/label_3.nii.gz']
        elif args.fold == 1:
            self.qry_list = ['./data/CHAOST2/chaos_MR_T2_normalized/image_10.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_15.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_19.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_8.nii.gz', ]
            self.qry_label_list = ['./data/CHAOST2/chaos_MR_T2_normalized/label_10.nii.gz',
                                   './data/CHAOST2/chaos_MR_T2_normalized/label_15.nii.gz',
                                   './data/CHAOST2/chaos_MR_T2_normalized/label_19.nii.gz',
                                   './data/CHAOST2/chaos_MR_T2_normalized/label_8.nii.gz', ]
            self.tmp_list = ['./data/CHAOST2/chaos_MR_T2_normalized/image_13.nii.gz']
            self.tmp_label_list = ['./data/CHAOST2/chaos_MR_T2_normalized/label_13.nii.gz']
        elif args.fold == 2:
            self.qry_list = ['./data/CHAOST2/chaos_MR_T2_normalized/image_20.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_22.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_19.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_31.nii.gz', ]
            self.qry_label_list = ['./data/CHAOST2/chaos_MR_T2_normalized/label_20.nii.gz',
                                   './data/CHAOST2/chaos_MR_T2_normalized/label_22.nii.gz',
                                   './data/CHAOST2/chaos_MR_T2_normalized/label_19.nii.gz',
                                   './data/CHAOST2/chaos_MR_T2_normalized/label_31.nii.gz', ]
            self.tmp_list = ['./data/CHAOST2/chaos_MR_T2_normalized/image_21.nii.gz']
            self.tmp_label_list = ['./data/CHAOST2/chaos_MR_T2_normalized/label_21.nii.gz']
        elif args.fold == 3:
            self.qry_list = ['./data/CHAOST2/chaos_MR_T2_normalized/image_36.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_34.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_32.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_31.nii.gz', ]
            self.qry_label_list = ['./data/CHAOST2/chaos_MR_T2_normalized/label_36.nii.gz',
                                   './data/CHAOST2/chaos_MR_T2_normalized/label_34.nii.gz',
                                   './data/CHAOST2/chaos_MR_T2_normalized/label_32.nii.gz',
                                   './data/CHAOST2/chaos_MR_T2_normalized/label_31.nii.gz', ]
            self.tmp_list = ['./data/CHAOST2/chaos_MR_T2_normalized/image_33.nii.gz']
            self.tmp_label_list = ['./data/CHAOST2/chaos_MR_T2_normalized/label_33.nii.gz']
        elif args.fold == 4:
            self.qry_list = ['./data/CHAOST2/chaos_MR_T2_normalized/image_1.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_36.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_38.nii.gz',
                             './data/CHAOST2/chaos_MR_T2_normalized/image_39.nii.gz', ]
            self.qry_label_list = ['./data/CHAOST2/chaos_MR_T2_normalized/label_1.nii.gz',
                                   './data/CHAOST2/chaos_MR_T2_normalized/label_36.nii.gz',
                                   './data/CHAOST2/chaos_MR_T2_normalized/label_38.nii.gz',
                                   './data/CHAOST2/chaos_MR_T2_normalized/label_39.nii.gz', ]
            self.tmp_list = ['./data/CHAOST2/chaos_MR_T2_normalized/image_37.nii.gz']
            self.tmp_label_list = ['./data/CHAOST2/chaos_MR_T2_normalized/label_37.nii.gz']



        # This is for cross-domain referring word-btcv
        # tmp_list = f'./data/Test_organ/word/support_img_list.txt'
        # with open(tmp_list, 'r') as f:
        #     self.tmp_list = [line.strip() for line in f.readlines()]
        # tmp_label_list = f'./data/Test_organ/word/support_lb_list.txt'
        # with open(tmp_label_list, 'r') as f:
        #     self.tmp_label_list = [line.strip() for line in f.readlines()]
        # self.organ_map = {1: 2, 2: 4, 3: 3, 4: 6, 5: 7, 6: 1, 7: 5, 11: 8, 12: 12, 13: 13}

        # This is for cross-domain referring btcv-word
        # tmp_list = f'./data/Test_organ/btcv/support_img_list.txt'
        # with open(tmp_list, 'r') as f:
        #     self.tmp_list = [line.strip() for line in f.readlines()]
        # tmp_label_list = f'./data/Test_organ/btcv/support_lb_list.txt'
        # with open(tmp_label_list, 'r') as f:
        #     self.tmp_label_list = [line.strip() for line in f.readlines()]
        # self.organ_map = {2: 1, 4: 2, 3: 3, 6: 4, 7: 5, 1: 6, 5: 7, 8: 11, 12: 12, 13: 13}

        # # This is for cross-domain referring t1in-ct
        # tmp_list = f'./data/Test_organ/CHAOS_MRI/support_img_list.txt'
        # with open(tmp_list, 'r') as f:
        #     self.tmp_list = [line.strip() for line in f.readlines()]
        # tmp_label_list = f'./data/Test_organ/CHAOS_MRI/support_lb_list.txt'
        # with open(tmp_label_list, 'r') as f:
        #     self.tmp_label_list = [line.strip() for line in f.readlines()]

        # This is for cross-domain referring ct-t1in
        # tmp_list = f'./data/Test_organ/chaos-ct/support_img_list.txt'
        # with open(tmp_list, 'r') as f:
        #     self.tmp_list = [line.strip() for line in f.readlines()]
        # tmp_label_list = f'./data/Test_organ/chaos-ct/support_lb_list.txt'
        # with open(tmp_label_list, 'r') as f:
        #     self.tmp_label_list = [line.strip() for line in f.readlines()]


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
            image_data_pre = vol
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

            if len(z_index) == 0:
                print('this support volume has no target object')
                continue

            starting_frame_nonzero = z_index[0]
            end_frame_nonzero = z_index[-1]
            hasobj_num_frame = end_frame_nonzero - starting_frame_nonzero + 1
            # tmp_indexes = list(range(starting_frame_nonzero, end_frame_nonzero + 1))
            tmp_indexes = list(range(gt.shape[-1]))
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
        # if self.dataset in ['T2SPIR', 'T1DUAL_InPhase', 'T1DUAL_OutPhase']:
        #     img_path = self.qry_list[index]
        #     img_path = img_path.replace('T1DUAL_InPhase', self.dataset)
        #     mask_path = img_path.replace('mri_', 'segmentation_')
        #     name = img_path.split('/')[-1].split('.')[0]
        # else:
        img_path = self.qry_list[index]
        mask_path = self.qry_label_list[index]
        name = img_path.split('/')[-1].split('.')[0]
        vol = nib.as_closest_canonical(nib.load(img_path)).get_fdata()  # h,w,d
        num_frames = vol.shape[-1]
        # clip intensity
        # if self.lower_bound == 'percentile_clip':
        #     lower_bound = np.percentile(vol, 0.5)
        #     upper_bound = np.percentile(vol, 99.5)
        #     # 使用这些分位数进行裁剪
        #     image_data_pre = np.clip(vol, lower_bound, upper_bound).astype(np.float32)
        # elif self.lower_bound is None:
        #     image_data_pre = vol
        # else:
        #     image_data_pre = np.clip(vol, self.lower_bound, self.upper_bound).astype(np.float32)  # 明确数据类型
        image_data_pre = vol
        # 计算一次min和max，避免重复计算
        min_val = np.min(image_data_pre)
        max_val = np.max(image_data_pre)
        data_3d = ((image_data_pre - min_val) / (max_val - min_val) * 255.0)

        gt = nib.as_closest_canonical(nib.load(mask_path)).get_fdata()
        gt[gt != int(self.organ)] = 0
        gt[gt == int(self.organ)] = 1
        gt[gt > 0] = 1
        _, _, z_index = np.where(gt == 1)
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


def enhance_contrast_CLAHE(image_numpy):
    """使用 CLAHE 自适应直方图均衡化增强图像对比度"""

    img_yuv = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2YUV)  # 转换到 YUV 颜色空间
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 创建 CLAHE 处理器
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])  # 仅对亮度通道增强
    enhanced_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)  # 转换回 RGB

    return enhanced_image  # 转换回 Tensor


class SUNSegTest(Dataset):
    def __init__(self, args, dataset, data_path, organ, support_num, transform=None, mode='Test',
                 prompt='click',
                 seed=None,
                 variation=0):

        qry_img_path = data_path
        qry_gt_path = qry_img_path.replace('img', 'gt')
        self.qry_list = [os.path.join(qry_img_path, case) for case in os.listdir(qry_img_path)]
        self.qry_label_list = [os.path.join(qry_gt_path, case) for case in os.listdir(qry_img_path)]
        group = dataset.split("_")[-1]
        # tmp_img_path = f'./data/Train_organ/SUN_SEG/imagesTr_{group}'
        tmp_img_path = f'./data/Train_organ/SUN_SEG/imagesTr'
        self.tmp_list = [
            os.path.join(tmp_img_path, case) for
            case in os.listdir(tmp_img_path)]

        self.tmp_label_list = [
            path.replace('imagesTr', 'labelsTr') for
            path in self.tmp_list]

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
        tmp_img_dirs = self.tmp_list  # 存储不同 Case 的图像文件夹路径
        tmp_mask_dirs = self.tmp_label_list  # 存储不同 Case 的 Mask 文件夹路径
        self.tmp_names = [os.path.basename(name) for name in tmp_img_dirs]

        sup_imgs = []
        sup_msks = []
        mid_stack_indices = []  # 存储 stack 后的中间 slice 索引
        cumulative_count = 0  # 记录已添加的 slice 数量

        # 遍历每个 Case
        index_dict = {}
        for i, (img_dir, mask_dir) in enumerate(zip(tmp_img_dirs, tmp_mask_dirs)):
            if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
                print(f"Skipping {img_dir} or {mask_dir}, directory not found.")
                continue

            # 获取当前 Case 下的所有 JPG 和 PNG 文件，并排序
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
            mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

            if len(img_files) == 0 or len(mask_files) == 0:
                print(f"Skipping {img_dir}, no valid JPG or PNG files found.")
            else:
                # 获取 mask 文件的无扩展名集合
                mask_names = {os.path.splitext(mask)[0] for mask in mask_files}

            # 仅选择能匹配上 mask 的 img 文件
            matched_files = [(img, f"{os.path.splitext(img)[0]}.png") for img in img_files if
                             os.path.splitext(img)[0] in mask_names]
            # 读取所有切片
            case_imgs = []
            case_msks = []
            # import ipdb
            # ipdb.set_trace()
            file_num = 0
            for img_file, mask_file in matched_files[::3]:
                file_num += 1
                img_path = os.path.join(img_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_file)
                # 读取 Mask
                mask = Image.open(mask_path).convert("L")  # 读取 mask 并转换为灰度
                mask = np.array(mask)
                mask = (mask - mask.min()) / (mask.max() - mask.min())
                mask_array = np.array(mask, dtype=np.uint8)  # (H, W)
                mask_array[mask_array != int(self.organ)] = 0
                mask_array[mask_array == int(self.organ)] = 1  # 只保留目标 organ
                assert np.sum(mask_array) != 0

                # 读取 RGB 图像
                img = Image.open(img_path).convert("RGB")
                img_array = np.array(img, dtype=np.uint8)  # (H, W, 3)
                img_array = enhance_contrast_CLAHE(img_array)

                # 归一化处理

                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255.0

                # 调整图像尺寸
                img_resized = cv2.resize(img_array, newsize, interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask_array, newsize, interpolation=cv2.INTER_NEAREST)

                # 转换为 Torch Tensor
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1)  # (C, H, W)
                mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0)  # (1, H, W)

                case_imgs.append(img_tensor)
                case_msks.append(mask_tensor)

            # 计算中间 slice 索引
            mid_stack_index = file_num // 2
            mid_stack_indices.append(cumulative_count + mid_stack_index)
            index_dict[img_dir] = [0, cumulative_count]

            # recent_masks = torch.stack(case_msks)  # shape: (file_num, 1, H, W)
            # foreground_areas = recent_masks.sum(dim=(1, 2, 3))  # shape: (file_num,)
            # max_foreground_index = torch.argmax(foreground_areas).item()
            # mid_stack_indices.append(cumulative_count + max_foreground_index)

            # 累加 slice 计数
            cumulative_count += file_num

            # 添加到总列表
            sup_imgs.extend(case_imgs)
            sup_msks.extend(case_msks)

        # 堆叠所有 slices
        sup_imgs = torch.stack(sup_imgs, dim=0)  # (K, C, H, W)
        sup_msks = torch.stack(sup_msks, dim=0)  # (K, 1, H, W)

        return sup_imgs, sup_msks, mid_stack_indices, index_dict

    def __len__(self):
        return len(self.qry_list)

    def __getitem__(self, index):

        newsize = (self.img_size, self.img_size)

        """Get the query images"""
        img_dir = self.qry_list[index]  # 存储的是 Case 目录，而不是 .nii.gz 文件
        mask_dir = self.qry_label_list[index]  # 存储的是 Mask 目录
        name = os.path.basename(img_dir)  # 取 Case 名称

        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            print(f"Skipping {img_dir} or {mask_dir}, directory not found.")
            return None

        # 读取所有 JPG 和 PNG，并确保文件名匹配
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

        if len(img_files) == 0 or len(mask_files) == 0:
            print(f"Skipping {img_dir}, no valid JPG or PNG files found.")
        else:
            # 获取 mask 文件的无扩展名集合
            mask_names = {os.path.splitext(mask)[0] for mask in mask_files}

        # 仅选择能匹配上 mask 的 img 文件
        matched_files = [(img, f"{os.path.splitext(img)[0]}.png") for img in img_files if
                         os.path.splitext(img)[0] in mask_names]

        if len(matched_files) != len(img_files):
            print(f"Warning: Some files in {img_dir} and {mask_dir} do not match exactly! Skipping this case.")
            return None

        num_frames = len(matched_files)

        # 读取所有切片
        img_arr = np.zeros((*newsize, 3, num_frames), dtype=np.float32)  # (H, W, 3, D)
        label_arr = np.zeros((*newsize, num_frames), dtype=np.float32)  # (H, W, D)
        for i, (img_file, mask_file) in enumerate(matched_files):
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            # 读取 Mask
            mask = Image.open(mask_path).convert("L")  # 读取 mask 并转换为灰度
            mask = np.array(mask)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask_array = np.array(mask, dtype=np.uint8)  # (H, W)
            mask_array[mask_array != int(self.organ)] = 0
            mask_array[mask_array == int(self.organ)] = 1  # 只保留目标 organ

            # 读取 RGB 图像
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img, dtype=np.uint8)  # (H, W, 3)
            img_array = enhance_contrast_CLAHE(img_array)
            # 归一化处理
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255.0

            # 调整图像尺寸
            img_resized = cv2.resize(img_array, newsize, interpolation=cv2.INTER_LINEAR)  # (H, W, 3)
            mask_resized = cv2.resize(mask_array, newsize, interpolation=cv2.INTER_NEAREST)  # (H, W)

            # 存入数组
            img_arr[:, :, :, i] = img_resized  # (H, W, 3)
            label_arr[:, :, i] = mask_resized  # (H, W)

        # 计算中间 Slice 作为 `prompt_id`
        _, _, z_index = np.where(label_arr > 0)
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

        if len(z_index) != 0:
            # 返回最大外接框
            # 在所有维度上投影,找到非零位置
            rows = np.any(label_arr, axis=(1, 2))  # 在y,z维度投影到x轴
            cols = np.any(label_arr, axis=(0, 2))  # 在x,z维度投影到y轴

            # 获取极值坐标
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            hw_max_bbox = [x_min, y_min, x_max, y_max]
        else:
            hw_max_bbox = [0, 0, 0, 0]

        # 转换为 PyTorch Tensor
        img_tensor = torch.from_numpy(img_arr.astype(np.float32))  # (H, W, 3, D)
        img_tensor = img_tensor.permute(3, 2, 0, 1)  # (D, 3, H, W)

        label_tensor = torch.from_numpy(label_arr).long()  # (H, W, D)
        label_tensor = label_tensor.permute(2, 0, 1)  # (D, H, W)

        # 生成 Click & BBox 提示
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}  # {'frame_idx': coordinates}

        for i in range(num_frames):
            mask = label_tensor[i].numpy()
            if self.prompt == 'click':
                point_label_dict[i], pt_dict[i] = random_click(mask, 1, seed=None)
            if self.prompt == 'bbox':
                bbox_dict[i] = generate_bbox(mask, variation=self.variation, seed=self.seed)

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
        elif self.prompt == 'None' or self.prompt == 'medlsam':
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
