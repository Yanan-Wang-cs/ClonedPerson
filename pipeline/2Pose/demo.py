from mmpose.apis import (inference_top_down_pose_model, init_pose_model)
import os.path
import cv2
import torch
import numpy as np
import csv
import shutil
import time

from models.data_util import process_mmdet_results, pose_image

my_start = time.time()
device = torch.device('cuda:0')
pose_model = init_pose_model(
    './models/hrnet_w48_coco_256x192.py',
        './models/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth',
    device=device)


imgFolder = 'img/'  # image folder
resultFolder = 'result/'  # result folder
pointsFile = "pedestron_points.csv" # keypoint position csv
saveRoot = './show'

if not os.path.exists(saveRoot):
    os.makedirs(saveRoot)

def classfication_according_to_keypoints(keypoint, img):
    # Create folders
    if not os.path.exists(resultFolder):
        os.makedirs(resultFolder)
        os.makedirs(resultFolder + '/back')
        os.makedirs(resultFolder + '/side')
        os.makedirs(resultFolder + '/occlude')
        os.makedirs(resultFolder + '/abnormal')
        os.makedirs(resultFolder + '/keypoint_qualified')
    keypoint = np.array(list(map(float, keypoint))).reshape(-1, 3).astype(int)

    # Body areas
    areaTop = np.array([keypoint[6], keypoint[5], keypoint[11], keypoint[12]])[:, 0:2]
    areaBottom = np.array([keypoint[12], keypoint[11], keypoint[13], keypoint[14]])[:, 0:2]
    # Expand ratio
    alpha = 0.1
    # Expand areas
    topDis0 = areaTop[1][0] - areaTop[0][0]
    areaTop[0][0] -= alpha * topDis0
    areaTop[1][0] += alpha * topDis0
    areaTop[2][0] -= alpha * topDis0
    areaTop[3][0] += alpha * topDis0

    bottomDis0 = areaBottom[1][0] - areaBottom[0][0]
    areaBottom[0][0] -= alpha * bottomDis0
    areaBottom[1][0] += alpha * bottomDis0
    areaBottom[2][0] -= alpha * bottomDis0
    areaBottom[3][0] += alpha * bottomDis0
    # points of hands and elbows
    dot1 = (int(keypoint[8][0]), int(keypoint[8][1]))
    dot2 = (int(keypoint[10][0]), int(keypoint[10][1]))
    dot3 = (int(keypoint[7][0]), int(keypoint[7][1]))
    dot4 = (int(keypoint[9][0]), int(keypoint[9][1]))
    # rule1
    if keypoint[5][0] > keypoint[6][0]:
        W = np.linalg.norm(keypoint[5, 0:2] - keypoint[6, 0:2])
        H = np.linalg.norm(keypoint[12, 0:2] - keypoint[6, 0:2])
        # rule2
        if W > 0 and H > 0 and W / H < 0.3:
            shutil.move(img, resultFolder + '/side/')
        elif W > 0 and H > 0 and W / H >= 0.3:
            # rule3
            if cv2.pointPolygonTest(areaTop, dot1, False) < 0 and cv2.pointPolygonTest(areaTop, dot2,False) < 0 and cv2.pointPolygonTest(areaTop, dot3, False) < 0 and cv2.pointPolygonTest(areaTop, dot4,False) < 0 and cv2.pointPolygonTest(areaBottom,dot1,False) < 0 and cv2.pointPolygonTest(areaBottom, dot2, False) < 0 and cv2.pointPolygonTest(areaBottom, dot3,False) < 0 and cv2.pointPolygonTest(areaBottom, dot4, False) < 0:
                shutil.move(img, resultFolder + '/keypoint_qualified/')
            else:
                shutil.move(img, resultFolder + '/occlude/')
        else:
            shutil.move(img, resultFolder + '/abnormal/')
    else:
        shutil.move(img, resultFolder + '/back/')


with open(pointsFile,"w") as csvfile:
    writer = csv.writer(csvfile)
    for img in os.listdir(imgFolder):
        try:
            image = cv2.imread(imgFolder+ img)
            person_results = []
            person = {}
            # bbox format is 'xywh'
            person['bbox'] = [0,0, image.shape[1], image.shape[0], 0.9]
            person_results.append(person)
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                image,
                person_results,
                bbox_thr=None,
                format='xyxy',
                return_heatmap=False,
                outputs=None)

            arr = pose_results[0]['keypoints'].reshape(-1)
            writer.writerows([[resultFolder+img,','.join(str(x) for x in arr)]])
            classfication_according_to_keypoints(arr, imgFolder+img)
            #image_tmp = pose_image(image, pose_results[0])
            #cv2.imwrite(os.path.join(saveRoot, img), image_tmp)
        
        except:
            print(img)
my_end = time.time()
print('running time: ', my_end-my_start)

