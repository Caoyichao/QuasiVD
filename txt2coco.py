# -*- coding: utf-8 -*-

import shutil
import os
import argparse
import glob
import random
import time
import torch
import cv2
import json

######将第一张图像从各个子文件夹中抽取出来#########
'''
root = "E:\\python\\Two-frame-detection\\Data"
random.seed(42)
neg_folder = 'Neg'
pos_folder = 'Pos'
ratio = 0.95
neg_samples = [os.path.join(root, neg_folder, f) for f in os.listdir(os.path.join(root, neg_folder))]
pos_samples = [os.path.join(root, pos_folder, f) for f in os.listdir(os.path.join(root, pos_folder))]
samples = neg_samples + pos_samples
random.shuffle(samples)
number = len(samples)
node = int(number * ratio)
train_samples = samples[:node]
val_samples = samples[node:]

for train_img in train_samples:
    src_path_split = train_img.split("\\")
    src_path = os.path.join(train_img,src_path_split[-1]+"-0001.txt")
    det_path = os.path.join(root,"train_smoke\\",src_path_split[-1]+"-0001.txt")
    os.system ("copy %s %s" % (src_path, det_path))

for val_img in val_samples:
    src_path_split = val_img.split("\\")
    src_path = os.path.join(val_img,src_path_split[-1]+"-0001.txt")
    det_path = os.path.join(root,"val_smoke\\",src_path_split[-1]+"-0001.txt")
    os.system ("copy %s %s" % (src_path, det_path))
'''
#####将图片名称重命名为数字####
'''
root = "E:\\python\\Two-frame-detection\\Data\\rename\\val_smoke"
newroot = "E:\\python\\Two-frame-detection\\Data\\rename\\v"
files = os.listdir(root)
i = 10000
for file in files:
    if ".jpg" in file:
        i += 1
        imgpath = os.path.join(root,file)
        txtpath = imgpath.replace("jpg","txt")
        num_str = ('{:0>5d}'.format(i))
        print(num_str)
        new_imgpath =  os.path.join(newroot,num_str+".jpg")
        new_txtpath =  os.path.join(newroot,num_str+".txt")
        os.system ("copy %s %s" % (imgpath, new_imgpath))
        os.system ("copy %s %s" % (txtpath, new_txtpath))
'''
#########json文件生成############
#---------------------------------------------------------------------------------------------------------

id_counter = 0 # To record the id
FILE_PATH = 'E:\\python\\Two-frame-detection\\Data\\rename\\v' #####
out = {'annotations': [], 
           'categories': [{"id": 0, "name": "firesmoke", "supercategory": ""}], ##### change the categories to match your dataset!
           'images': [],
           'info': {"contributor": "", "year": "", "version": "", "url": "", "description": "", "date_created": ""},
           'licenses': {"id": 0, "name": "", "url": ""}
           }

def annotations_data(whole_path , image_id):
    # id, bbox, iscrowd, image_id, category_id
    global id_counter
    txt = open(whole_path,'r')
    for line in txt.readlines(): # if txt.readlines is null, this for loop would not run
        data = line.strip()
        data = data.split()
        # convert the center into the top-left point!
        #data[1] = float(data[1])* 800 - 0.5 * float(data[3])* 800 ##### change the 800 to your raw image width
        #data[2] = float(data[2])* 600 - 0.5 * float(data[4])* 600 ##### change the 600 to your raw image height
        #data[3] = float(data[3])* 800 ##### change the 800 to your raw image width
        #data[4] = float(data[4])* 600 ##### change the 600 to your raw image height
        bbox = [data[0],data[1],data[2],data[3]]
        ann = {'id':int(image_id),
            'bbox': bbox,
            'area': int(data[3]) * int(data[4]),
            'iscrowd': 0,
            'image_id': image_id,
            'category_id': int(0)           
        }
        out['annotations'].append(ann)
        id_counter = id_counter + 1 

def images_data(img_path,file_name):
    #id, height, width, file_name
    id = file_name.split('.')[0]
    img_data = cv2.imread(img_path)
    file_name = id + '.jpg' ##### change '.jpg' to other image formats if the format of your image is not .jpg
    imgs = {'id': int(id),
            'height': img_data.shape[1], ##### change the 600 to your raw image height
            'width': img_data.shape[0], ##### change the 800 to your raw image width
            'file_name': file_name,
            "coco_url": "", 
            "flickr_url": "", 
            "date_captured": 0, 
            "license": 0
    }
    out['images'].append(imgs)


if __name__ == '__main__':
    files = os.listdir(FILE_PATH)
    files.sort()
    for file in files:
        if "txt" in file:
            whole_path = os.path.join(FILE_PATH,file)
            txt_path = whole_path
            annotations_data(whole_path, file.split('.')[0])
            img_path = whole_path.replace("txt","jpg")
            images_data(img_path,file)

    
    with open('instances_val_smoke.json', 'w') as outfile: ##### change the str to the json file name you want
      json.dump(out, outfile, separators=(',', ':'))
