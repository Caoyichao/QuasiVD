import os
import cv2
import numpy as np
import pandas as pd
import math
import random
import torch.utils.data as data
from sklearn.externals import joblib

from image import flip, color_aug
from image import get_affine_transform, affine_transform
from image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian

class CTDetDataset(data.Dataset):
    default_resolution = [512, 512]
    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __init__(self, opt, split):
        super(CTDetDataset, self).__init__()
        self.num_classes = opt.num_class
        self.max_objs = 20
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.opt = opt
        self.split = split
        if split == 'train':
            self.file_list = opt.train
        elif split == 'val':
            self.file_list = opt.val
        else:
            ValueError('not valid split!')

        random_forest_model = 'random_forest.model'
        self.random_forest = joblib.load(random_forest_model)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def get_mask(self, image, anns):
        shape = image.shape[:2]
        mask = np.zeros(shape, dtype=np.uint8)
        for ann in anns:
            ann = [i if i >= 0 else 0 for i in ann]
            patch = image[ann[1]: ann[3], ann[0]: ann[2]]
            patch_shape = patch.shape
            patch = patch.reshape((-1, 3))
            df = pd.DataFrame({'B': patch[:, 0], 'G': patch[:, 1], 'R': patch[:, 2]})
            pred = self.random_forest.predict_proba(df)
            pred = pred[:, 1]
            pred[pred < 0.5] = 0
            pred = (pred * 255).astype(np.uint8)
            pred = pred.reshape((patch_shape[0], patch_shape[1]))
            pred = cv2.erode(pred, self.kernel)
            pred = cv2.dilate(pred, self.kernel)
            mask[ann[1]: ann[3], ann[0]: ann[2]] = pred
        return mask

    def __getitem__(self, index):
        '''
            img_info['image_path']： 序列帧的目录
            img_info['label'] ： 序列帧对应的标签
        '''
        img_info = self.file_list[index]
        img_path = img_info['image_path']
        tick = len(img_path) // 2      # 取2帧做序列检测
        start_num = random.randint(1, tick) - 1
        img_1_path = img_path[start_num]
        img_2_path = img_path[start_num + tick]

        anns_all = img_info['label']
        anns = anns_all[start_num]
        num_objs = min(len(anns), self.max_objs)

        if 'Pos' in img_1_path:
            cls = 1
        else:
            cls = 0

        image_1 = cv2.imread(img_1_path)
        # image_2 = cv2.imread(img_2_path)

        image_1_float = cv2.imread(img_1_path, 0).astype(np.float32)
        image_2_float = cv2.imread(img_2_path, 0).astype(np.float32)
        image_2 = abs(image_2_float - image_1_float)
        image_2 = image_2 / np.max(image_2)

        shape = image_1.shape[:2]
        mask = np.zeros(shape, dtype=np.uint8)
        if 'Pos' in img_1_path:
            mask = self.get_mask(image_1, anns)

        height, width = image_1.shape[0], image_1.shape[1]
        c = np.array([image_1.shape[1] / 2., image_1.shape[0] / 2.], dtype=np.float32)

        s = max(image_1.shape[0], image_1.shape[1]) * 1.0
        input_h, input_w = self.default_resolution[0], self.default_resolution[1]

        flipped = False
        if self.split == 'train':
            s = s * np.random.choice(np.arange(0.6, 1., 0.1))
            w_border = self._get_border(500, image_1.shape[1])
            h_border = self._get_border(300, image_1.shape[0])
            c[0] = np.random.randint(low=w_border, high=image_1.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=image_1.shape[0] - h_border)

            if np.random.random() < self.opt.flip:
                flipped = True
                image_1 = image_1[:, ::-1, :]
                image_2 = image_2[:, ::-1]
                mask = mask[:, ::-1]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp_1 = cv2.warpAffine(image_1, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp_2 = cv2.warpAffine(image_2, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (input_w//4, input_h//4))
        inp_2 = cv2.resize(inp_2, (input_w//4, input_h//4))

        # debug
        # src_img = inp_1.copy()

        inp_1 = (inp_1.astype(np.float32) / 255.)
        # inp_2 = (inp_2.astype(np.float32) / 255.)
        mask = (mask.astype(np.float32) / 255.)
        if self.split == 'train':
            color_aug(self._data_rng, inp_1, self._eig_val, self._eig_vec)
            # color_aug(self._data_rng, inp_2, self._eig_val, self._eig_vec)
        inp_1 = (inp_1 - self.mean) / self.std
        # inp_2 = (inp_2 - self.mean) / self.std
        inp_1 = inp_1.transpose(2, 0, 1)
        # inp_2 = inp_2.transpose(2, 0, 1)


        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        draw_gaussian = draw_umich_gaussian

        pos = []
        for k in range(num_objs):
            bbox = np.array(anns[k][:4], dtype=np.float32)
            cls_id = anns[k][-1]
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                pos.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

        inp_2 = np.expand_dims(inp_2, axis=0)
        mask = np.expand_dims(mask, axis=0) * 0.9 + hm * 0.1   # best (0.9 & 0.1: 85.93)
        mask = mask / (np.max(mask) + 1e-9)
        # mask = cv2.GaussianBlur(mask, (5, 5), 0)

        ret = {'input1': inp_1, 'input2': inp_2, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
               'reg': reg, 'cls': cls, 'mask': mask}

        # debug
        '''
        if num_objs:
            src_img = cv2.resize(src_img, (128, 128))
            heatmap = mask[0].copy()
            heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / np.max(heatmap)
            cam = heatmap + np.float32(src_img/255)
            cam = cam / np.max(cam)
            cam = np.uint8(255 * cam)
            # cam = cv2.GaussianBlur(cam, (5, 5), 0)
            for item in pos:
                cam = cv2.rectangle(cam, (item[0], item[1]), (item[2], item[3]), (255, 0, 0), 1)
            base_name = os.path.basename(img_1_path)
            cv2.imwrite(os.path.join('vis_output', base_name), cam)
        '''

        return ret

    def __len__(self):
        return len(self.file_list)