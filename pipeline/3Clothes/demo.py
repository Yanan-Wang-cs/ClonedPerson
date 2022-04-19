import cv2, os
import sys


import importlib
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from PIPNet_rmnb.lib.networks import *
# import data_utils
from PIPNet_rmnb.lib.functions import *
from mmcv import Config
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv.image import imread, imwrite
import os
import numpy as np
import cv2
import csv
import shutil
import time

my_start = time.time()

imgFolder = 'img/'
clothPointDress = open('DF_dress.csv', 'w') # This file saves the dress points
clothPointPair = open('DF_pair.csv', 'w') # This file saves the outfit poitns


saveFolder = 'outfit_qualified'
unqualifiedFolder = 'outfit_unqualified'
unknowTypeFolder = 'outfit_unknowtype'
if not os.path.exists(imgFolder + '/'+saveFolder):
    os.makedirs(imgFolder + '/'+saveFolder)
    os.makedirs(imgFolder + '/' + unqualifiedFolder)
    os.makedirs(imgFolder + '/' + unknowTypeFolder)

my_thresh = 0.6
det_box_scale = 1.2

# config
data_name = 'Clothes_1'
experiment_name_Clothes_1 = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
save_dir_Clothes_1 = os.path.join('./PIPNet_rmnb/snapshots', data_name, experiment_name_Clothes_1)

resnet18_Clothes_1 = models.resnet18(pretrained=False)
net_Clothes_1 = Pip_resnet18(resnet18_Clothes_1, num_lms=19, input_size=256, net_stride=32)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_Clothes_1 = net_Clothes_1.to(device)
weight_file_Clothes_1 = os.path.join(save_dir_Clothes_1, 'epoch%d.pth' % (59))
state_dict_Clothes_1 = torch.load(weight_file_Clothes_1, map_location=device)
net_Clothes_1.load_state_dict(state_dict_Clothes_1)



data_name = 'Clothes_2'
experiment_name_Clothes_2 = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
save_dir_Clothes_2 = os.path.join('./PIPNet_rmnb/snapshots', data_name, experiment_name_Clothes_2)
resnet18_Clothes_2 = models.resnet18(pretrained=True)
net_Clothes_2 = Pip_resnet18(resnet18_Clothes_2, num_lms=15, input_size=256, net_stride=32)
net_Clothes_2 = net_Clothes_2.to(device)
weight_file_Clothes_2 = os.path.join(save_dir_Clothes_2, 'epoch%d.pth' % (59))
state_dict_Clothes_2 = torch.load(weight_file_Clothes_2, map_location=device)
net_Clothes_2.load_state_dict(state_dict_Clothes_2)



data_name = 'Clothes_3'
experiment_name_Clothes_3 = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
save_dir_Clothes_3 = os.path.join('./PIPNet_rmnb/snapshots', data_name, experiment_name_Clothes_3)
resnet18_Clothes_3 = models.resnet18(pretrained=True)
net_Clothes_3 = Pip_resnet18(resnet18_Clothes_3, num_lms=11, input_size=256, net_stride=32)
net_Clothes_3 = net_Clothes_3.to(device)
weight_file_Clothes_3 = os.path.join(save_dir_Clothes_3, 'epoch%d.pth' % (59))
state_dict_Clothes_3 = torch.load(weight_file_Clothes_3, map_location=device)
net_Clothes_3.load_state_dict(state_dict_Clothes_3)



data_name = 'Clothes_4'
experiment_name_Clothes_4 = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
save_dir_Clothes_4 = os.path.join('./PIPNet_rmnb/snapshots', data_name, experiment_name_Clothes_4)
resnet18_Clothes_4 = models.resnet18(pretrained=True)
net_Clothes_4 = Pip_resnet18(resnet18_Clothes_4, num_lms=14, input_size=256, net_stride=32)
net_Clothes_4 = net_Clothes_4.to(device)
weight_file_Clothes_4 = os.path.join(save_dir_Clothes_4, 'epoch%d.pth' % (59))
state_dict_Clothes_4 = torch.load(weight_file_Clothes_4, map_location=device)
net_Clothes_4.load_state_dict(state_dict_Clothes_4)


data_name = 'Clothes_5'
experiment_name_Clothes_5 = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
save_dir_Clothes_5 = os.path.join('./PIPNet_rmnb/snapshots', data_name, experiment_name_Clothes_5)
resnet18_Clothes_5 = models.resnet18(pretrained=True)
net_Clothes_5 = Pip_resnet18(resnet18_Clothes_5, num_lms=10, input_size=256, net_stride=32)
net_Clothes_5 = net_Clothes_5.to(device)
weight_file_Clothes_5 = os.path.join(save_dir_Clothes_5, 'epoch%d.pth' % (59))
state_dict_Clothes_5 = torch.load(weight_file_Clothes_5, map_location=device)
net_Clothes_5.load_state_dict(state_dict_Clothes_5)

#
data_name = 'Clothes_8'
experiment_name_Clothes_8 = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
save_dir_Clothes_8 = os.path.join('./PIPNet_rmnb/snapshots', data_name, experiment_name_Clothes_8)
resnet18_Clothes_8 = models.resnet18(pretrained=True)
net_Clothes_8 = Pip_resnet18(resnet18_Clothes_8, num_lms=7, input_size=256, net_stride=32)
net_Clothes_8 = net_Clothes_8.to(device)
weight_file_Clothes_8 = os.path.join(save_dir_Clothes_8, 'epoch%d.pth' % (59))
state_dict_Clothes_8 = torch.load(weight_file_Clothes_8, map_location=device)
net_Clothes_8.load_state_dict(state_dict_Clothes_8)


data_name = 'Clothes_13'
experiment_name_Clothes_13 = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
save_dir_Clothes_13 = os.path.join('./PIPNet_rmnb/snapshots', data_name, experiment_name_Clothes_13)
resnet18_Clothes_13 = models.resnet18(pretrained=True)
net_Clothes_13 = Pip_resnet18(resnet18_Clothes_13, num_lms=22, input_size=256, net_stride=32)
net_Clothes_13 = net_Clothes_13.to(device)
weight_file_Clothes_13 = os.path.join(save_dir_Clothes_13, 'epoch%d.pth' % (59))
state_dict_Clothes_13 = torch.load(weight_file_Clothes_13, map_location=device)
net_Clothes_13.load_state_dict(state_dict_Clothes_13)


data_name = 'Clothes_17'
experiment_name_Clothes_17 = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
save_dir_Clothes_17 = os.path.join('./PIPNet_rmnb/snapshots', data_name, experiment_name_Clothes_17)
resnet18_Clothes_17 = models.resnet18(pretrained=True)
net_Clothes_17 = Pip_resnet18(resnet18_Clothes_17, num_lms=16, input_size=256, net_stride=32)
net_Clothes_17 = net_Clothes_17.to(device)
weight_file_Clothes_17 = os.path.join(save_dir_Clothes_17, 'epoch%d.pth' % (59))
state_dict_Clothes_17 = torch.load(weight_file_Clothes_17, map_location=device)
net_Clothes_17.load_state_dict(state_dict_Clothes_17)



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), normalize])
#

map_dict_id = dict({'0': 2, '1': 1, '4': 3, '6': 5, '7': 4, '8': 8, '9': 13, '11': 17})
map_dict_landmarks = dict({'1': 19, '0': 15, '4': 11, '7': 14, '6': 10, '8': 7, '9': 22, '11': 16})
# map_dict_net = dict({'1':net_Clothes_1})

map_dict_net = dict({'1':net_Clothes_1,  '0':net_Clothes_2, '4':net_Clothes_3, '7':net_Clothes_4,
                     '6':net_Clothes_5, '8':net_Clothes_8, '9':net_Clothes_13, '11':net_Clothes_17})



cfg = Config.fromfile('./mmdetection/configs_old/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco.py')
cfg.model.roi_head.bbox_head.num_classes = 13

# Setup a checkpoint file to load
checkpoint = './mmdetection/deppFashion2_multigpu/epoch_12.pth'
# checkpoint = 'cloth_detection/epoch_12.pth'
# initialize the detector
model = init_detector(cfg, checkpoint, device='cuda:0')
model.CLASSES = ['short_sleeve_top', 'long_sleeve_top','short_sleeve_outwear', 'long_sleeve_outwear', 'vest',
              'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress', 'long_sleeve_dress', 'vest_dress',
               'sling_dress']



def detect_landmarks(net, images):
    det_crop = images
    # det_crop = cv2.resize(images, (256, 256))
    inputs = Image.fromarray(det_crop[:, :, ::-1].astype('uint8'), 'RGB')
    inputs = preprocess(inputs).unsqueeze(0)
    inputs = inputs.to(device)
    lms_pred_x, lms_pred_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, 256, 32)
    lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
    lms_pred = lms_pred.cpu().numpy()

    return lms_pred
#
#

for img in os.listdir(imgFolder):
    if 'png' in img or 'jpg' in img or 'jpeg' in img:
        write_content = []
        try:
            result = inference_detector(model, os.path.join(imgFolder, img))
            bboxes = np.vstack(result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(result)
            ]
            labels = np.concatenate(labels)

            assert bboxes.ndim == 2
            assert labels.ndim == 1
            assert bboxes.shape[0] == labels.shape[0]
            assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

            scores = bboxes[:, -1]

            # my_thresh
            score_tmp = my_thresh
            inds = scores >= score_tmp
            bboxes1 = bboxes[inds, :]
            labels1 = labels[inds]
            scores = scores[inds]

            # suitable type
            inds = [i for i in range(len(labels1)) if labels1[i] in [0,1,4,6,7,8,9,11]]
            bboxes1 = bboxes1[inds, :]
            labels1 = labels1[inds]
            scores = scores[inds]
            types = [int(map_dict_id[str(p)]) for p in labels1]

            inds = np.argsort(types)
            bboxes1 = bboxes1[inds, :]
            labels1 = labels1[inds]
            scores = scores[inds]
            types = np.array(types)[inds]

            final_label=[]
            final_bboxes=[]
            top = []
            bottom = []
            dress = []

            # clothes classification
            if len(labels1)>0:
                for p in range(len(labels1)-1, -1, -1):
                    if types[p] > 9:
                        if p == len(labels1)-1:
                            final_label.append(labels1[p])
                            final_bboxes.append(bboxes1[p])
                            dress.append(labels1[p])
                    elif types[p] > 3:
                        if len(bottom) == 0 and len(dress) == 0:
                            final_label.append(labels1[p])
                            final_bboxes.append(bboxes1[p])
                            bottom.append(labels1[p])
                    else:
                        if len(top) == 0 and len(dress) == 0:
                            final_label.append(labels1[p])
                            final_bboxes.append(bboxes1[p])
                            top.append(labels1[p])

                # clothes keypoints detection
                img1 = imread(os.path.join(imgFolder, img))
                image_height, image_width, _ = img1.shape
                # for bbox, label in zip(bboxes, labels):

                for q in range(0, len(final_label)):
                    labels = final_label[q]
                    bboxes = final_bboxes[q]
                    bbox = bboxes
                    label = labels

                    if not str(label) in map_dict_id:
                        continue

                    bbox_int = bbox.astype(np.int32)
    
                    det_xmin = bbox_int[0]
                    det_ymin = bbox_int[1]
                    det_xmax = bbox_int[2]
                    det_ymax = bbox_int[3]
                    det_width = det_xmax - det_xmin
                    det_height = det_ymax - det_ymin

                    det_xmin -= int(det_width * (det_box_scale - 1) / 2)
                    # remove a part of top area for alignment, see paper for details
                    det_ymin -= int(det_height * (det_box_scale - 1) / 2)
                    det_xmax += int(det_width * (det_box_scale - 1) / 2)
                    det_ymax += int(det_height * (det_box_scale - 1) / 2)
                    det_xmin = max(det_xmin, 0)
                    det_ymin = max(det_ymin, 0)
                    det_xmax = min(det_xmax, image_width - 1)
                    det_ymax = min(det_ymax, image_height - 1)
                    det_width = det_xmax - det_xmin
                    det_height = det_ymax - det_ymin
                    # # cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
                    det_crop = img1[det_ymin:det_ymax, det_xmin:det_xmax, :]
    
                    ladmarks = detect_landmarks(map_dict_net[str(label)], det_crop)

                    # ladmark_results = []
                    pretext = ''
                    if len(dress) > 0:
                        txt_file_choose = clothPointDress
                        pretext = 'dress_'
                    elif len(bottom) > 0 and len(top)>0:
                        txt_file_choose = clothPointPair
                        pretext = 'pair_'
                    else:
                        continue
                    txt_file_choose.write(os.path.join(saveFolder, pretext+img) + ',' + str(map_dict_id[str(label)]) + ',' + '"')

                    for i in range(map_dict_landmarks[str(label)]):
                        x_pred = ladmarks[i * 2] * det_width + det_xmin
                        y_pred = ladmarks[i * 2 + 1] * det_height + det_ymin
                        # cv2.circle(img1, (int(x_pred), int(y_pred)), 1, (0, 0, 255), 2)
    
                        if i < map_dict_landmarks[str(label)] -1:
                            txt_file_choose.write(str(int(x_pred)) + ',' + str(int(y_pred)) + ',')
                        else:
                            txt_file_choose.write(str(int(x_pred)) + ',' + str(int(y_pred)) + '"')
    
                    txt_file_choose.write(','+ str(0)+ '\n')
                if len(dress) > 0:
                    shutil.move(imgFolder + img, imgFolder + saveFolder + '/dress_' + img)
                elif len(bottom) > 0 and len(top) > 0:
                    shutil.move(imgFolder + img, imgFolder + saveFolder + '/pair_' + img)
                else:
                    shutil.move(imgFolder + img, imgFolder + unqualifiedFolder +'/' + img)
            else:
                shutil.move(imgFolder + img, imgFolder + unknowTypeFolder +'/' + img)
        except:
            print('wrong!!!!!!!!!!',img)
clothPointDress.close()
clothPointPair.close()
my_end=time.time()
print('running time: ', my_end-my_start)


