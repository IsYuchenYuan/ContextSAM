""" Dataloader for the BTCV dataset
    Yunli Qi
"""

from PIL import Image
from func_3d.utils import random_click, generate_bbox
import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import pickle
import PIL.Image
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import cv2

def read_image(path):
    with open(path, 'rb') as file:
        img = pickle.load(file)
        return img


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k, axes=(0, 1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-15, 15)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def convert_to_PIL(img: np.array) -> PIL.Image:
    '''
    img should be normalized between 0 and 1
    '''
    img = np.clip(img, 0, 1)
    return PIL.Image.fromarray((img * 255).astype(np.uint8))


def convert_to_PIL_label(label):
    return PIL.Image.fromarray(label.astype(np.uint8))


def convert_to_np(img: PIL.Image) -> np.array:
    return np.array(img).astype(np.float32) / 255


def convert_to_np_label(label):
    return np.array(label).astype(np.float32)


def random_erasing(
        imgs,
        label,
        scale_z=(0.02, 0.33),
        scale=(0.02, 0.05),
        ratio=(0.3, 3.3),
        apply_all: int = 0,
        rng: np.random.Generator = np.random.default_rng(0),
):
    # determine the box
    imgshape = imgs.shape

    # nx and ny
    while True:
        se = rng.uniform(scale[0], scale[1]) * imgshape[0] * imgshape[1]
        re = rng.uniform(ratio[0], ratio[1])
        nx = int(np.sqrt(se * re))
        ny = int(np.sqrt(se / re))
        if nx < imgshape[1] and ny < imgshape[0]:
            break

    # determine the position of the box
    sy = rng.integers(0, imgshape[0] - ny + 1)
    sx = rng.integers(0, imgshape[1] - nx + 1)

    # print(nz, ny, nx, sz, sy, sx)
    filling = rng.uniform(0, 1, size=[ny, nx])
    filling = filling[:, :, np.newaxis]
    filling = np.repeat(filling, imgshape[-1], axis=-1)

    # erase
    imgs[sy:sy + ny, sx:sx + nx, :] = filling
    label[sy:sy + ny, sx:sx + nx, :] = 0.

    return imgs, label


def posterize(img, label, v):
    '''
    4 < v < 8
    '''
    v = int(v)

    for slice_indx in range(img.shape[2]):
        img_curr = img[:, :, slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageOps.posterize(img_curr, bits=v)
        img_curr = convert_to_np(img_curr)
        img[:, :, slice_indx] = img_curr

    return img, label


def contrast(img, label, v):
    '''
    0.1 < v < 1.9
    '''

    for slice_indx in range(img.shape[2]):
        img_curr = img[:, :, slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageEnhance.Contrast(img_curr).enhance(v)
        img_curr = convert_to_np(img_curr)
        img[:, :, slice_indx] = img_curr

    return img, label


def brightness(img, label, v):
    '''
    0.1 < v < 1.9
    '''

    for slice_indx in range(img.shape[2]):
        img_curr = img[:, :, slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageEnhance.Brightness(img_curr).enhance(v)
        img_curr = convert_to_np(img_curr)
        img[:, :, slice_indx] = img_curr

    return img, label


def sharpness(img, label, v):
    '''
    0.1 < v < 1.9
    '''
    for slice_indx in range(img.shape[2]):
        img_curr = img[:, :, slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageEnhance.Sharpness(img_curr).enhance(v)
        img_curr = convert_to_np(img_curr)
        img[:, :, slice_indx] = img_curr

    return img, label


def identity(img, label, v):
    return img, label


def adjust_light(image, label):
    image = image * 255.0
    gamma = random.random() * 3 + 0.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    for slice_indx in range(image.shape[2]):
        img_curr = image[:, :, slice_indx]
        img_curr = cv2.LUT(np.array(img_curr).astype(np.uint8), table).astype(np.uint8)
        image[:, :, slice_indx] = img_curr
    image = image / 255.0

    return image, label


def shear_x(img, label, v):
    '''
    -0.3 < v < 0.3
    '''
    shear_mat = [1, v, -v * img.shape[1] / 2, 0, 1, 0]  # center the transform

    for slice_indx in range(img.shape[2]):
        img_curr = img[:, :, slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:, :, slice_indx] = img_curr
        # print(img.shape)

    for slice_indx in range(label.shape[2]):
        label_curr = label[:, :, slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:, :, slice_indx] = label_curr
        # print(label.shape)

    return img, label


def shear_y(img, label, v):
    '''
    -0.3 < v < 0.3
    '''
    shear_mat = [1, 0, 0, v, 1, -v * img.shape[0] / 2]  # center the transform

    for slice_indx in range(img.shape[2]):
        img_curr = img[:, :, slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:, :, slice_indx] = img_curr
        # print(img.shape)

    for slice_indx in range(label.shape[2]):
        label_curr = label[:, :, slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:, :, slice_indx] = label_curr
        # print(label.shape)

    return img, label


def translate_x(img, label, v):
    '''
    -0.45 < v < 0.45
    '''
    translate_mat = [1, 0, v * img.shape[1], 0, 1, 0]

    for slice_indx in range(img.shape[2]):
        img_curr = img[:, :, slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:, :, slice_indx] = img_curr

    for slice_indx in range(label.shape[2]):
        label_curr = label[:, :, slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:, :, slice_indx] = label_curr

    return img, label


def translate_y(img, label, v):
    '''
    -0.45 < v < 0.45
    '''
    translate_mat = [1, 0, 0, 0, 1, v * img.shape[0]]

    for slice_indx in range(img.shape[2]):
        img_curr = img[:, :, slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:, :, slice_indx] = img_curr

    for slice_indx in range(label.shape[2]):
        label_curr = label[:, :, slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:, :, slice_indx] = label_curr

    return img, label


def scale(img, label, v):
    '''
    0.6 < v < 1.4
    '''
    for slice_indx in range(img.shape[2]):
        img_curr = img[:, :, slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, [v, 0, 0, 0, v, 0], resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:, :, slice_indx] = img_curr

    for slice_indx in range(label.shape[2]):
        label_curr = label[:, :, slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, [v, 0, 0, 0, v, 0],
                                          resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:, :, slice_indx] = label_curr

    return img, label

# class RandomGenerator(object):
#     def __init__(self, output_size, aug=False):
#         self.output_size = output_size
#         self.aug = aug
#         seed = 42
#         self.rng = np.random.default_rng(seed)
#         self.p = 0.5
#         self.n = 1
#         self.scale = (0.8, 1.2, 2)
#         self.translate = (-0.2, 0.2, 2)
#         self.shear = (-0.3, 0.3, 2)
#         self.posterize = (4, 8.99, 2)
#         self.contrast = (0.7, 1.3, 2)
#         self.brightness = (0.7, 1.3, 2)
#         self.sharpness = (0.1, 1.9, 2)
#
#         self.create_ops()
#
#     def create_ops(self):
#         ops = [
#             # (shear_x, self.shear),
#             # (shear_y, self.shear),
#             # (scale, self.scale),
#             # (translate_x, self.translate),
#             # (translate_y, self.translate),
#             # (posterize, self.posterize),
#             (contrast, self.contrast),
#             (brightness, self.brightness),
#             (sharpness, self.sharpness),
#             (identity, (0, 1, 1)),
#         ]
#
#         self.ops = [op for op in ops if op[1][2] != 0]
#
#
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#
#         if self.aug:
#             # if random.random() > 0.5:
#             #     image, label = random_rot_flip(image, label)
#             # if random.random() > 0.5:
#             #     image, label = random_rotate(image, label)
#             # if random.random() > 0.5:
#             #     image, label = adjust_light(image, label)
#             # if random.random() > 0.5:
#             #     inds = self.rng.choice(len(self.ops), size=self.n, replace=False)
#             #     for i in inds:
#             #         op = self.ops[i]
#             #         aug_func = op[0]
#             #         aug_params = op[1]
#             #         v = self.rng.uniform(aug_params[0], aug_params[1])
#             #         image, label = aug_func(image, label, v)
#
#             op = self.ops[1]
#             aug_func = op[0]
#             aug_params = op[1]
#             v = self.rng.uniform(aug_params[0], aug_params[1])
#             image, label = aug_func(image, label, v)
#
#
#         x, y, z = image.shape
#         if x != self.output_size[0] or y != self.output_size[1]:
#             image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=3)
#             label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=0)
#         image = torch.from_numpy(image.astype(np.float32))
#         label = torch.from_numpy(label.astype(np.float32))
#         image = image.permute(2, 0, 1)  # d,h,w
#         label = label.permute(2, 0, 1)
#
#         sample = {'image': image, 'label': label.long()}
#         return sample

"""by yuchen"""

def change_intensity(image, factor):
    # 调整强度，factor>1 增强，factor<1 减弱
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def adjust_contrast(image, factor):
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

def add_gaussian_noise(image, std=10):
    noise = np.random.normal(0, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image.astype(np.uint8), lut)

def enhance_slices(volume, enhance_function, *args):
    enhanced_volume = np.empty_like(volume)
    for i in range(volume.shape[2]):
        slice_ = volume[:, :, i]
        enhanced_volume[:, :, i] = enhance_function(slice_, *args)
    return enhanced_volume

class RandomGenerator(object):
    def __init__(self, output_size, aug=False):
        self.output_size = output_size
        self.aug = aug
        seed = 42
        self.rng = np.random.default_rng(seed)
        self.p = 0.5
        self.n = 2
        self.contrast = (0.7, 1.3, 2)
        self.intensity = (0.7, 1.3, 2)
        self.std = (10, 15, 2)
        self.gamma = (0.7, 1.3, 2)

        self.create_ops()


    def create_ops(self):
        ops = [
            (change_intensity, self.intensity),
            (adjust_contrast, self.contrast),
            (add_gaussian_noise, self.std),
            (gamma_correction, self.gamma),

        ]

        self.ops = [op for op in ops if op[1][2] != 0]

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.aug:
            if random.random() > 0.5:
                inds = self.rng.choice(len(self.ops), size=self.n, replace=False)
                for i in inds:
                    op = self.ops[i]
                    aug_func = op[0]
                    aug_params = op[1]
                    v = self.rng.uniform(aug_params[0], aug_params[1])
                    image = enhance_slices(image, aug_func, v)
        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        image = image.permute(2, 0, 1)  # d,h,w
        label = label.permute(2, 0, 1)

        sample = {'image': image, 'label': label.long()}
        return sample


class BTCV(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'image'))

        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'mask', name)
        data_seg_3d_shape = np.load(mask_path + '/0.npy').shape
        num_frame = len(os.listdir(mask_path))
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,))
        for i in range(num_frame):
            data_seg_3d[..., i] = np.load(os.path.join(mask_path, f'{i}.npy'))
        for i in range(data_seg_3d.shape[-1]):
            if np.sum(data_seg_3d[..., i]) > 0:
                data_seg_3d = data_seg_3d[..., i:]
                break
        starting_frame_nonzero = i
        for j in reversed(range(data_seg_3d.shape[-1])):
            if np.sum(data_seg_3d[..., j]) > 0:
                data_seg_3d = data_seg_3d[..., :j + 1]
                break
        num_frame = data_seg_3d.shape[-1]
        if self.video_length is None:
            video_length = int(num_frame / 4)
        else:
            video_length = self.video_length
        if num_frame > video_length and self.mode == 'Training':
            starting_frame = np.random.randint(0, num_frame - video_length + 1)
        else:
            starting_frame = 0
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}  # {'frame_idx',{'obj',mask(h,w)}}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}  # {'frame_idx',{'obj',coordinates}}

        for frame_index in range(starting_frame, starting_frame + video_length):
            img = Image.open(os.path.join(img_path, f'{frame_index + starting_frame_nonzero}.jpg')).convert('RGB')
            mask = data_seg_3d[..., frame_index]
            # mask = np.rot90(mask)
            obj_list = np.unique(mask[mask > 0])
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                # if self.transform_msk:
                obj_mask = Image.fromarray(obj_mask)
                obj_mask = obj_mask.resize(newsize)
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                # obj_mask = self.transform_msk(obj_mask).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)),
                                                                                         point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation,
                                                            seed=self.seed)
            # if self.transform:
            # state = torch.get_rng_state()
            # img = self.transform(img)
            # torch.set_rng_state(state)
            img = img.resize(newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)  # 3,h,w

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
            }


class BTCV_Spt(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'image'))

        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None
        self.obj_groups = [(4, 6), (2, 3), (8, 9), (5, 8, 9), (6, 10, 11), (7,), (1,), (12, 13), (1, 6, 7)]
        self.support_set = [f'./data/BTCV_clip/Training/image/img0025/57.jpg',
                            f'./data/BTCV_clip/Training/image/img0025/46.jpg',
                            f'./data/BTCV_clip/Training/image/img0025/61.jpg',
                            f'./data/BTCV_clip/Training/image/img0025/78.jpg',
                            f'./data/BTCV_clip/Training/image/img0025/60.jpg',
                            f'./data/BTCV_clip/Training/image/img0025/57.jpg',
                            f'./data/BTCV_clip/Training/image/img0025/67.jpg',
                            f'./data/BTCV_clip/Training/image/img0025/60.jpg',
                            f'./data/BTCV_clip/Training/image/img0025/69.jpg']

    def __len__(self):
        if self.mode == 'Training':
            return len(self.name_list)
        elif self.mode == 'Test' or self.mode == 'Testing':
            return len(self.name_list) * len(self.obj_groups)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)
        """Get the template images"""
        # 随机选择分组
        if self.mode == 'Training':
            group_index = random.choice(range(len(self.obj_groups)))
        elif self.mode == 'Test' or self.mode == 'Testing':
            group_index = index % len(self.obj_groups)

        """固定template set"""
        # sup_img_path = self.support_set[group_index]
        # sup_mask_path = sup_img_path.replace("image", "mask").replace("jpg", "npy")
        # sup_img = Image.open(sup_img_path).convert('RGB')
        # sup_img = sup_img.resize(newsize)
        # sup_img = torch.tensor(np.array(sup_img)).permute(2, 0, 1)  # 3,h,w
        # sup_mask = np.load(sup_mask_path)  # h,w
        # sup_mask = Image.fromarray(sup_mask)
        # sup_mask = sup_mask.resize(newsize, Image.NEAREST)
        # sup_mask = torch.from_numpy(np.array(sup_mask)).unsqueeze(0)  # 1,h,w

        """template从其他volume中随机采样"""
        random_index = random.choice(range(len(self.name_list)))
        while random_index == index:
            random_index = random.choice(range(len(self.name_list)))
        sup_case = self.name_list[random_index]
        img_path = os.path.join(self.data_path, self.mode, 'image', sup_case)
        mask_path = os.path.join(self.data_path, self.mode, 'mask', sup_case)
        data_seg_3d_shape = np.load(mask_path + '/0.npy').shape
        num_frame = len(os.listdir(mask_path))
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,))
        for idx in range(num_frame):
            data_seg_3d[..., idx] = np.load(os.path.join(mask_path, f'{idx}.npy'))
        group_mask = np.isin(data_seg_3d, np.array(list(self.obj_groups[group_index])))
        group_mask = group_mask.transpose(2, 0, 1) # d,h,w
        indices = np.nonzero(group_mask)
        starting_frame_nonzero = indices[0][0]
        end_frame_nonzero = indices[0][-1]
        hasobj_num_frame = end_frame_nonzero - starting_frame_nonzero + 1
        sup_idx = starting_frame_nonzero + hasobj_num_frame // 2
        sup_img_path = os.path.join(img_path, f'{sup_idx}.jpg')
        sup_mask_path = sup_img_path.replace("image", "mask").replace("jpg", "npy")
        sup_img = Image.open(sup_img_path).convert('RGB')
        sup_img = sup_img.resize(newsize)
        sup_img = torch.tensor(np.array(sup_img)).permute(2, 0, 1)  # 3,h,w
        sup_mask = np.load(sup_mask_path)  # h,w
        sup_mask = Image.fromarray(sup_mask)
        sup_mask = sup_mask.resize(newsize, Image.NEAREST)
        sup_mask = torch.from_numpy(np.array(sup_mask)).unsqueeze(0)  # 1,h,w

        """Get the query clip"""
        if self.mode == 'Training':
            name = self.name_list[index]
        elif self.mode == 'Test' or self.mode == 'Testing':
            index = index // len(self.obj_groups)
            name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'mask', name)
        data_seg_3d_shape = np.load(mask_path + '/0.npy').shape
        num_frame = len(os.listdir(mask_path))
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,))
        for idx in range(num_frame):
            data_seg_3d[..., idx] = np.load(os.path.join(mask_path, f'{idx}.npy'))
        group_mask = np.isin(data_seg_3d, np.array(list(self.obj_groups[group_index])))
        data_seg_3d[~group_mask] = 0  # h,w,d
        data_seg_3d_depth = data_seg_3d.transpose(2, 0, 1)
        indices = np.nonzero(data_seg_3d_depth)
        starting_frame_nonzero = indices[0][0]
        end_frame_nonzero = indices[0][-1]
        hasobj_num_frame = end_frame_nonzero - starting_frame_nonzero + 1

        if self.video_length is None:
            video_length = int(num_frame / 4)
        else:
            video_length = self.video_length

        if random.random() < 0.3:
            if hasobj_num_frame > video_length:
                starting_frame = np.random.randint(starting_frame_nonzero, end_frame_nonzero - video_length + 1)
            else:
                end_frame = min(starting_frame_nonzero + video_length, num_frame)
                starting_frame = end_frame - video_length
        else:
            # print("%%%%%%%%%%%%%%%%%sample empty%%%%%%%%")
            if random.random() < 0.5:
                mid = np.random.randint(starting_frame_nonzero, starting_frame_nonzero + video_length // 2)
                starting_frame = max(0, mid - video_length // 2)
            else:
                mid = np.random.randint(end_frame_nonzero - video_length // 2 + 1, end_frame_nonzero)
                end_frame = min(mid + video_length // 2, num_frame - 1)
                starting_frame = end_frame - video_length + 1

        img_arr = np.zeros((data_seg_3d_shape[0], data_seg_3d_shape[1], video_length))  # h,w,d
        label_arr = np.zeros((data_seg_3d_shape[0], data_seg_3d_shape[1], video_length))  # h,w,d
        for frame_index in range(starting_frame, starting_frame + video_length):
            img = Image.open(os.path.join(img_path, f'{frame_index}.jpg')).convert('L')
            assert len(np.array(img).shape) == 2
            mask = data_seg_3d[..., frame_index]
            img_arr[:, :, frame_index - starting_frame] = np.float32(img)  # h,w
            label_arr[:, :, frame_index - starting_frame] = np.float32(mask)
        contain_obj = np.zeros((video_length, len(self.obj_groups[group_index])))
        # 遍历每个深度切片
        for d in range(video_length):
            for i, obj in enumerate(self.obj_groups[group_index]):
                # 检查当前切片是否包含标签 1
                contain_obj[d, i] = (label_arr[:, :, d] == obj).any()
        sample = {'image': img_arr, 'label': label_arr}
        if self.transform:
            sample = self.transform(sample)
        img_tensor = sample['image']  # d,h,w: float()
        img_tensor = img_tensor.unsqueeze(1).repeat(1, 3, 1, 1)  # d,3,h,w
        label_tensor = sample['label']  # d,h,w: long()

        mask_dict = {}  # {'frame_idx',{'obj',mask(h,w)}}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}  # {'frame_idx',{'obj',coordinates}}
        for i in range(video_length):
            mask = label_tensor[i].numpy()
            obj_list = np.unique(mask[mask > 0])
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()  # 1,h,w
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)),
                                                                                         point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation,
                                                            seed=self.seed)
            mask_dict[i] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[i] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[i] = diff_obj_pt_dict
                point_label_dict[i] = diff_obj_point_label_dict

        ann_obj_list = list(self.obj_groups[group_index])
        ann_obj_list = np.array(ann_obj_list).reshape(1, -1)  # 1,num
        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'sup_img': sup_img,
                'sup_mask': sup_mask,
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
                'ann_obj_list': ann_obj_list,
                'contain_obj': contain_obj
            }
        elif self.prompt == 'click':
            return {
                'sup_img': sup_img,
                'sup_mask': sup_mask,
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
                'ann_obj_list': np.array(ann_obj_list),
                'contain_obj': contain_obj
            }


class BTCVSingleCls(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None):

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'image'))

        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = args.variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None
        self.cls_to_slice = {
            1: 139,
            2: 125,
            3: 119,
            4: 146,
            5: 180,
            6: 154,
            7: 161,
            8: 147,
            9: 136,
            10: 145,
            11: 130,
            12: 140,
            13: 134,
        }

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'mask', name)
        data_seg_3d_shape = np.load(mask_path + '/0.npy').shape
        num_frame = len(os.listdir(mask_path))
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,))
        for i in range(num_frame):
            data_seg_3d[..., i] = np.load(os.path.join(mask_path, f'{i}.npy'))
        # 该vol包含的所有物体类比
        obj_list = np.unique(data_seg_3d[data_seg_3d > 0])
        # 随机选取一类进行训练
        select_cls = np.random.choice(obj_list, 1, replace=False)[0]
        """ Get template images"""
        t_slice = self.cls_to_slice[select_cls]
        sup_img_path = os.path.join(self.data_path, 'Test/image/img0003', f'{t_slice}.jpg')
        tmp_mask_path = os.path.join(self.data_path, 'Test/mask/img0003', f'{t_slice}.npy')
        sup_img = Image.open(sup_img_path).convert('RGB')  # h,w,c
        sup_img = sup_img.resize(newsize)
        sup_img = torch.tensor(np.array(sup_img)).permute(2, 0, 1).unsqueeze(0)  # 1,3,h,w
        sup_mask = np.load(tmp_mask_path)
        sup_mask[sup_mask != select_cls] = 0
        sup_mask[sup_mask == select_cls] = 1
        sup_mask = Image.fromarray(sup_mask)
        sup_mask = sup_mask.resize(newsize)
        sup_mask = torch.tensor(np.array(sup_mask)).unsqueeze(0).unsqueeze(0).int()  # 1,1,h,w
        # 转换成binary mask
        data_seg_3d[data_seg_3d != select_cls] = 0
        data_seg_3d[data_seg_3d == select_cls] = 1
        # 提取该类器官的前景区域
        for i in range(data_seg_3d.shape[-1]):
            if np.sum(data_seg_3d[..., i]) > 0:
                data_seg_3d = data_seg_3d[..., i:]
                break
        starting_frame_nonzero = i
        for j in reversed(range(data_seg_3d.shape[-1])):
            if np.sum(data_seg_3d[..., j]) > 0:
                data_seg_3d = data_seg_3d[..., :j + 1]
                break
        num_frame = data_seg_3d.shape[-1]
        if self.video_length is None:
            video_length = int(num_frame / 4)
        else:
            video_length = self.video_length
        if num_frame > video_length and self.mode == 'Training':
            starting_frame = np.random.randint(0, num_frame - video_length + 1)
        else:
            starting_frame = 0
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = torch.zeros(video_length, 1, self.img_size, self.img_size)
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}  # {'frame_idx',{'obj',coordinates}}

        for frame_index in range(starting_frame, starting_frame + video_length):
            img = Image.open(os.path.join(img_path, f'{frame_index + starting_frame_nonzero}.jpg')).convert('RGB')
            mask = data_seg_3d[..., frame_index]
            obj_mask = mask
            # if self.transform_msk:
            obj_mask = Image.fromarray(obj_mask)
            obj_mask = obj_mask.resize(newsize)
            obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()  # 1,h,w

            if self.prompt == 'click':
                diff_obj_point_label, diff_obj_pt = random_click(np.array(obj_mask.squeeze(0)),
                                                                 point_label, seed=None)
            elif self.prompt == 'bbox':
                diff_obj_bbox = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation,
                                              seed=self.seed)

            img = img.resize(newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)  # 3,h,w

            img_tensor[frame_index - starting_frame, :, :, :] = img  # d,3,h,w
            mask_dict[frame_index - starting_frame] = obj_mask  # d,1,h,w
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label

        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'sup_image': sup_img,
                'sup_mask': sup_mask,
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'sup_image': sup_img,
                'sup_mask': sup_mask,
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
            }


class BTCVTest(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'image'))

        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        # name = 'img0007'
        img_path = os.path.join(self.data_path, self.mode, 'image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'mask', name)
        data_seg_3d_shape = np.load(mask_path + '/0.npy').shape
        num_frame = len(os.listdir(mask_path))
        data_seg_3d = np.zeros(newsize + (num_frame,))
        for i in range(num_frame):
            mask = np.load(os.path.join(mask_path, f'{i}.npy'))
            mask = Image.fromarray(mask)
            mask = mask.resize(newsize, Image.NEAREST)  # 多种颜色插值记得设为nearest
            data_seg_3d[..., i] = np.array(mask)
        mask_tensor = torch.tensor(data_seg_3d).permute(2, 0, 1).unsqueeze(1).int()

        num_frame = data_seg_3d.shape[-1]
        if self.video_length is None:
            video_length = num_frame
        else:
            video_length = self.video_length

        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(video_length):
            img = Image.open(os.path.join(img_path, f'{frame_index}.jpg')).convert('RGB')
            mask = data_seg_3d[..., frame_index]
            obj_list = np.unique(mask[mask > 0])
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                obj = int(obj)
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)),
                                                                                         point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation,
                                                            seed=self.seed)
            img = img.resize(newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)  # c,h,w [0-255]
            img_tensor[frame_index, :, :, :] = img

            if self.prompt == 'bbox':
                bbox_dict[frame_index] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index] = diff_obj_pt_dict
                point_label_dict[frame_index] = diff_obj_point_label_dict

        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'img_path': img_path,
                'image': img_tensor,
                'label': mask_tensor,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
                # 'prompt_frame_idx': center_slices,
                # 'start_frame_idx': start_frame_idx
            }
        elif self.prompt == 'click':
            return {
                'img_path': img_path,
                'image': img_tensor,
                'label': mask_tensor,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
                # 'prompt_frame_idx': center_slices,
                # 'start_frame_idx': start_frame_idx
            }


class BTCVTest_GTPrompt(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'image'))
        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'mask', name)
        data_seg_3d_shape = np.load(mask_path + '/0.npy').shape
        num_frame = len(os.listdir(mask_path))
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,))
        for i in range(num_frame):
            data_seg_3d[..., i] = np.load(os.path.join(mask_path, f'{i}.npy'))

        num_frame = data_seg_3d.shape[-1]
        if self.video_length is None:
            video_length = num_frame
        else:
            video_length = self.video_length

        # find the middle slice of each object
        center_slices = {}  # {'prompt_idx':obj_id}
        obj_lists = list(np.unique(data_seg_3d[data_seg_3d > 0]))
        for class_id in obj_lists:
            slices_with_class = np.where(np.any(data_seg_3d == class_id, axis=(0, 1)))[0]
            # 计算最中间的 slice 索引
            middle_index = len(slices_with_class) // 2
            middle_slice = slices_with_class[middle_index]
            center_slices[int(class_id)] = middle_slice
        # center_slices = list(set(center_slices))
        mid_video_length = video_length // 2
        distance = {}
        # find the slice that is the closest to the middle of the 3d data to start prompting
        for slice in center_slices.values():
            distance[slice] = abs(mid_video_length - slice)
        start_frame_idx = sorted(distance, key=distance.get)[0]

        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(video_length):
            img = Image.open(os.path.join(img_path, f'{frame_index}.jpg')).convert('RGB')
            mask = data_seg_3d[..., frame_index]
            # mask = np.rot90(mask)
            obj_list = np.unique(mask[mask > 0])
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                obj = int(obj)
                # if self.transform_msk:
                obj_mask = Image.fromarray(obj_mask)
                obj_mask = obj_mask.resize(newsize)
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                # obj_mask = self.transform_msk(obj_mask).int()
                diff_obj_mask_dict[obj] = obj_mask  # 1,h,w

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)),
                                                                                         point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation,
                                                            seed=self.seed)

            img = img.resize(newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)  # c,h,w [0-255]

            img_tensor[frame_index, :, :, :] = img
            mask_dict[frame_index] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index] = diff_obj_pt_dict
                point_label_dict[frame_index] = diff_obj_point_label_dict

        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'img_path': img_path,
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
                'prompt_frame_idx': center_slices,
                'start_frame_idx': start_frame_idx
            }
        elif self.prompt == 'click':
            return {
                'img_path': img_path,
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
                'prompt_frame_idx': center_slices,
                'start_frame_idx': start_frame_idx
            }
