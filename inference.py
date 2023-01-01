# -*- coding: utf-8 -*
import os
import glob
import random
import cv2
import numpy as np
import tqdm
import json
import torch
import torch.nn.functional as F

from image import get_affine_transform
from processor import ctdet_post_process, ctdet_decode
from model import Model
from get_mobilev3 import FastDet


num_class = 1
thresh = 640
max_per_image = 10
is_submission = False
heads = {'hm': num_class, 'wh': 2, 'reg': 2}
arch = 'mobilenet'

if arch == 'FFNet':
    net = Model(num_class)
    net_path = './data/FFNet_best.pth.tar'
elif arch == 'mobilenet':
    net = FastDet(num_class, out_channels=80, output_shape=512 // 4, pretrained=None)
    net_path = './data/mobilenet_best-90.35..pth.tar'
else:
    raise ('No model!!')

checkpoint = torch.load(net_path)
# base_dict = checkpoint['state_dict']
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
net = net.cuda()
net.eval()


class Model():
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406],
                        dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225],
                       dtype=np.float32).reshape(1, 1, 3)
        self.scales = [1.0]


    def predict_test(self, datas):
        labels = []
        with torch.no_grad():
            for data_path in tqdm.tqdm(datas):
                img_path = sorted(glob.glob(data_path + '/*.jpg'))
                tick = len(img_path) // 2  # 取2帧做序列检测
                start_num = random.randint(1, tick) - 1
                img_1_path = img_path[start_num]
                img_2_path = img_path[start_num + tick]
                image_1 = cv2.imread(img_1_path)
                src_img = image_1.copy()
                src_shape = image_1.shape
                # image_2 = cv2.imread(img_2_path)

                image_1_float = cv2.imread(img_1_path, 0).astype(np.float32)
                image_2_float = cv2.imread(img_2_path, 0).astype(np.float32)
                image_2 = abs(image_2_float - image_1_float)
                image_2 = image_2 / np.max(image_2)

                # cls head
                '''
                image_1 = cv2.resize(image_1, (960, 640))
                image_2 = cv2.resize(image_2, (960, 640))
                inp_1 = (image_1.astype(np.float32) / 255.)
                inp_2 = (image_2.astype(np.float32) / 255.)
                inp_1 = inp_1.transpose(2, 0, 1)
                inp_2 = inp_2.transpose(2, 0, 1)
                images = np.concatenate((inp_1, inp_2), axis=0)
                images = torch.from_numpy(images).unsqueeze(0).cuda()
                logits, mask = net(images)
                h_x = F.softmax(logits, 1)[0]
                conf, pred_label = torch.max(h_x, 0)
                mask = mask[0][0].cpu().detach().numpy()
                mask = np.array(mask * 255).astype(np.uint8)
                mask_path = 'vis_output/' + os.path.basename(img_2_path)[:-4] + '_{}_{}_mask.png'.format(float('%.3f' % conf), int(pred_label))
                os.system('cp -r {} {}'.format(img_2_path, 'vis_output'))
                cv2.imwrite(mask_path, mask)
                '''

                # det head
                detections = []
                image_1, meta, scale = self.pre_process(image_1, 1.)
                image_1 = (image_1 / 255. - self.mean) / self.std
                image_1 = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0).float()
                image_2, meta, scale = self.pre_process(image_2, 1.)
                shape = image_1.shape[2:]
                image_2 = cv2.resize(image_2, (shape[1]//4, shape[0]//4))
                image_2 = torch.from_numpy(image_2)
                image_2 = image_2.unsqueeze(0).unsqueeze(0)

                image_1 = image_1.cuda()
                image_2 = image_2.cuda()
                # images = torch.cat([image_1, image_2], dim=1)
                dets, mask = self.process(image_1, image_2)

                mask = mask[0][0].cpu().detach().numpy()
                mask = np.array(mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (src_shape[1], src_shape[0]))
                mask_path = 'vis_output/' + os.path.basename(img_1_path)[:-4] + '_mask.png'
                #os.system('cp -r {} {}'.format(img_1_path, 'vis_output'))
                cv2.imwrite(mask_path, mask)
                #continue

                dets = self.post_process(dets, meta, scale)
                detections.append(dets)
                results = self.merge_outputs(detections)


                label = []
                for cat, data in results.items():
                    data_label = np.full((data.shape[0], 1), cat)
                    data = np.hstack((data_label, data))
                    label.append(data)
                data = np.vstack([lab for lab in label])
                score = data[:, -1]
                indices = np.argsort(score)[::-1]
                data = np.round(data[:, :-1]).astype('int64')

                data = np.hstack((data, score.reshape(-1, 1)))
                data = data[indices]
                data = np.clip(data, 0, 10000)
                #anchors_nms_idx = nms(
                #    data[:, 1:], thresh=0.5)
                #labels.append(data[anchors_nms_idx])
                
                for index in range(data.shape[0]):  #[cls,左上x,左上y，右下x，右下y，conf]
                    pos = [int(data[index][i]) for i in range(6)]
                    conf = "%.2f" % data[index][5]
                    if(float(conf)>0.2):
                        src_img = cv2.rectangle(src_img, (pos[1], pos[2]),(pos[3], pos[4]), (0, 0, 255), 2)
                img_res_path = 'vis_output/' + os.path.basename(img_1_path)[:-4] + '.jpg'
                cv2.imwrite(img_res_path,src_img)

            return labels


    def process(self, img1, img2):
        output, mask = net(img1, img2)
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']

        dets = ctdet_decode(hm, wh, reg=reg, K=10)
        return dets, mask


    def pre_process(self, image, scale):
        height, width = image.shape[0:2]
        max_val = max(height, width)
        if max_val >= thresh:     # constrain max image size within 1000
            scale = min(thresh / height, thresh / width)

        new_height = int(height * scale)
        new_width = int(width * scale)

        inp_height = (new_height | 127) + 1
        inp_width = (new_width | 127) + 1

        c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)


        # inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
        # inp_image = (inp_image / 255.).astype(np.float32)

        # images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

        # images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // 4,
                'out_width': inp_width // 4}
        return inp_image, meta, scale

    def post_process(self, dets, meta, scale=1.):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], num_class)

        for j in range(0, num_class):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(0, num_class):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1:
                nms(results[j], thresh=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(num_class)])
        if len(scores) > max_per_image:
            kth = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(0, num_class):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

def nms(dets, thresh, method=None):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-5)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def display(data_path, detection):
    img = cv2.imread(data_path)
    color_map = {0: (255,0,0), 1: (0,0,255)}
    for item in detection:
        cv2.rectangle(img, (item[1], item[2]), (item[3], item[4]), color_map[item[1]], 2)
    cv2.imwrite('vis.jpg', img)

def create_map_files(labels, datas):
    save_path = '../code/detect_result/'
    for i, label in enumerate(labels):
        info = []
        img_name = os.path.basename(datas[i]).split('.')[0]
        for val in label:
            # cls, conf, x1, y1, x2, y2
            recode = ' '.join([str(int(val[0])), str(val[5]), str(int(val[1])), str(int(val[2])), str(int(val[3])), str(int(val[4]))])
            info.append(recode)

        save_file = os.path.join(save_path, img_name + '.txt')
        with open(save_file, 'w') as f:
            f.write('\n'.join(info))

def get_val(root):
    random.seed(42)
    neg_folder = 'Neg'
    pos_folder = 'Pos'
    ratio = 0#0.95
    neg_samples = [os.path.join(root, neg_folder, f) for f in os.listdir(os.path.join(root, neg_folder))]
    pos_samples = [os.path.join(root, pos_folder, f) for f in os.listdir(os.path.join(root, pos_folder))]
    samples = neg_samples + pos_samples
    random.shuffle(samples)
    number = len(samples)
    node = int(number * ratio)
    val_samples = samples[node:]
    return val_samples

if __name__ == '__main__':
    if is_submission:
        data_path = '../Data/SingleFrame/images/'
        datas = sorted(glob.glob(data_path + '*.jpg'))
    else:
        #datas = get_val('./data/Two_frame_data')
        datas = get_val('E:/python/Data/Two-frame')
    model = Model()

    labels = model.predict_test(datas)
    #create_map_files(labels, datas)
