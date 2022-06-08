import numpy as np
import math
import os
import cv2
import random
from PIL import Image, ImageDraw, ImageFont

# scene, camera, dealTime, 'Recordings'+dealTime+'_scene'+scene+'/camera'+camera+'.mp4'
processList = [[99,0,1],[99,1,1]]

def extractLabel(videoFolder, camera, timestep = 0.5):
    basicFolder = videoFolder+'/person-im'
    extractFile = basicFolder + '/deal_camera'+str(camera)+'.txt'
    if os.path.exists(extractFile):
        print('ExtractLabel already existed.')
        return

    file = open(extractFile, 'a+')
    position = np.loadtxt(open(basicFolder + '/pointsCamera'+str(camera)+'.txt',"rb"), delimiter=",",skiprows=0, dtype=str)

    current_frame = 0
    start_frame = math.floor(0)
    current_item0 = ''
    current_id = ''
    for item in position:
        if item[0]:
            current_item0 = item[0]
        if item[1]:
            current_id = item[1]
        if current_item0 == '':
            continue
        if math.floor(float(current_item0)) > start_frame:
            start_frame = math.floor(float(current_item0))
        if float(current_item0) > 0 and (abs(float(current_item0)-start_frame) < timestep or float(current_item0)-current_frame == 0.0):
            current_frame = float(current_item0)
            if abs(float(current_item0)-start_frame) < timestep:
                start_frame += timestep
            str1 = ','
            if not item[0]:
                item[0] = current_item0
            if not item[1]:
                item[1] = current_id
            file.write(str1.join(item)+'\n')
    print('Complete extractLabel')

def cropDataset(videoFolder, scene, camera):
    show = 1
    recordPoints = 0
    heightAdd = 0.15
    minwh = 0.3
    framerate = 30
    replace_score = 0.7
    eff_area = [0, 0, 1920, 1080]
    minheight = 40
    minwidth = 30
    min_zgap = 0.2

    video_path = videoFolder + '/camera' + str(camera) +'.mp4'
    txt_path = videoFolder + '/person-im/deal_camera' + str(camera) +'.txt'
    save_folder = videoFolder + '/images/camera'+str(camera)
    camera_text = '_s' + str(scene).zfill(2) + '_c' + str(camera).zfill(2) + '_f'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        print('Crop file already existed.')
        return

    data = np.genfromtxt(txt_path, dtype=float, delimiter=',')
    cap = cv2.VideoCapture(video_path)

    def iou(dts, gts):
        dts = np.asarray(dts)
        gts = np.asarray(gts)
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            gx1 = gt[0]
            gy1 = gt[1]
            gx2 = gt[2]
            gy2 = gt[3]
            garea = (gt[2] - gt[0]) * (gt[3] - gt[1])
            for i, dt in enumerate(dts):
                dx1 = dt[0]
                dy1 = dt[1]
                dx2 = dt[2]
                dy2 = dt[3]
                darea = (dt[2]-dt[0]) * (dt[3]-dt[1])
                unionw = min(dx2, gx2) - max(dx1, gx1)
                if unionw <= 0:
                    continue
                unionh = min(dy2, gy2) - max(dy1, gy1)
                if unionh <= 0:
                    continue
                t = unionw * unionh
                ious[i, j] = max(float(t) / garea, float(t) / darea)
        return ious

    def get_boundingbox(frame, pointArray):
        x = pointArray[:, 2]
        y = pointArray[:, 3]
        x = np.append(x, pointArray[0][5])
        y = np.append(y, pointArray[0][6])
        x = np.clip(x, 0, frame.shape[1])
        y = np.clip(y, 0, frame.shape[0])
        top = frame.shape[0] - np.max(y)
        bottom = frame.shape[0] - np.min(y)
        left = np.min(x)
        right = np.max(x)
        if top >= frame.shape[0] - 5 or left >= frame.shape[1] - 5 or right < 5 or bottom < 5:
            return np.array([0, 0, 0, 0])
        h = bottom - top
        left = left - 0.1 * h
        right = right + 0.1 * h
        top = top - heightAdd * h
        bottom = bottom + 0.05 * h
        if (right - left) / (bottom - top) > 0.5:
            return np.array([0, 0, 0, 0])
        if (right - left) / (bottom - top) < minwh:
            centerx = (left + right) / 2
            left = centerx - minwh / 2 * (bottom - top)
            right = centerx + minwh / 2 * (bottom - top)
        return np.array([int(left), int(top), int(right), int(bottom)])

    def cutpic(frame_array):
        cap.set(cv2.CAP_PROP_POS_FRAMES, math.floor(float(frame_array[0][0]) * framerate))
        ret, frame = cap.read()
        rectArray = np.empty(shape=[0, 6])
        rectArray_cache = np.empty(shape=[0, 6])
        finalArray = np.empty(shape=[0, 6])

        # Ignored area in videos
        background = np.empty(shape=[0, 6])
        if scene == 2 and int(camera) == 3:
            background = np.row_stack((background, np.array([1218, 531, 1303, 601, 0, 1000000], dtype=float)))
            background = np.row_stack((background, np.array([1452, 526, 1560, 673, 0, 1000000], dtype=float)))
            background = np.row_stack((background, np.array([1693, 367, 1824,942, 0, 1000000], dtype=float)))

        end_id = -1
        txt_arr = []
        while end_id + 1 < len(frame_array):
            current_id = frame_array[end_id + 1][1]
            id_points = np.where(frame_array[:, 1] == current_id)
            rect = get_boundingbox(frame, frame_array[id_points[0]])

            if (rect[2] - rect[0]) <= 0 or (rect[3] - rect[1]) <= 0:
                end_id = id_points[0][len(id_points[0]) - 1]
                continue
            if len(background) > 0:
                backIouArray = iou(np.array([rect]), background[:, 0:4])
                if len(backIouArray) > 0 and np.max(backIouArray) > 0.2:
                    end_id = id_points[0][len(id_points[0]) - 1]
                    continue

            iouArray = iou(np.array([rect]), rectArray_cache[:, 0:4])

            replace = 0
            rectArray_cache = np.row_stack((rectArray_cache,np.array([rect[0], rect[1], rect[2], rect[3], current_id, frame_array[end_id + 1][4]],dtype=float)))

            if len(iouArray[0]) > 0:
                pic_type = 0
                for i in range(len(iouArray[0])):
                    if iouArray[0][i] > replace_score:
                        replace += 1
                        if abs(rectArray_cache[i][5] - frame_array[end_id + 1][4]) < min_zgap:
                            pic_type = 1
                            rectArray[i] = np.array([0, 0, 0, 0, rectArray[i][4], rectArray[i][5]])
                        else:
                            if rectArray_cache[i][5] < frame_array[end_id + 1][4]:
                                pic_type = 2
                            else:
                                if pic_type == 0:
                                    pic_type = 3
                                rectArray[i] = np.array([0, 0, 0, 0, rectArray[i][4],rectArray[i][5]])
                if pic_type == 1 or pic_type == 2:
                    rectArray = np.row_stack((rectArray, np.array([0, 0, 0, 0, current_id, frame_array[end_id + 1][4]], dtype=float)))
                else:
                    if pic_type == 3:
                        rectArray = np.row_stack((rectArray, np.array([rect[0], rect[1], rect[2], rect[3], current_id, frame_array[end_id + 1][4]], dtype=float)))
                if replace < 1:
                    rectArray = np.row_stack((rectArray, np.array([rect[0], rect[1], rect[2], rect[3], current_id,frame_array[end_id + 1][4]], dtype=float)))
            else:
                rectArray = np.row_stack((rectArray, np.array([rect[0], rect[1], rect[2], rect[3], current_id, frame_array[end_id + 1][4]], dtype=float)))
            end_id = id_points[0][len(id_points[0]) - 1]

        for item in rectArray:
            if math.floor(item[2])-math.floor(item[0]) > minwidth and math.floor(item[3])-math.floor(item[1])> minheight and item[0]> eff_area[0] and item[1]> eff_area[1] and item[2]< eff_area[2] and item[3]<eff_area[3]:
                p_position = np.where(frame_array[:, 1] == item[4])
                point_array = frame_array[p_position[0]]
                image_name = str(math.floor(item[4])).zfill(6) + camera_text + str(math.floor(float(frame_array[0][0]) * framerate)).zfill(6) + '.jpg'
                finalArray = np.row_stack((finalArray, np.array(item, dtype=float)))
                if recordPoints:
                    txt = image_name+','+str(int(item[0]))+','+str(int(item[1]))+','+str(int(item[2]))+','+str(int(item[3]))
                    for i in range(7):
                        index = np.where(point_array[:, 7] == i+1)
                        x = int(point_array[index[0]][0][2] - item[0])
                        y = int(frame.shape[0] - point_array[index[0]][0][3] - item[1])
                        if show > 0:
                            cv2.circle(frame, (int(item[0]) + x, int(item[1] + y)), 2, (255, 255, 0), 10)
                        txt += ','+str(x)+','+str(y)
                    if txt not in txt_arr:
                        txt_arr.append(txt)
                cv2.imwrite(save_folder + "/" + image_name,frame[math.floor(item[1]):math.floor(item[3]), math.floor(item[0]):math.floor(item[2])])

        if recordPoints:
            with open(save_folder+"_point.txt", "a", encoding="utf-8") as f2:
                for item in txt_arr:
                    f2.write(item+'\n')
        if show > 0 and len(rectArray)>0:
            for item in rectArray_cache:
                # All labeled bounding boxes
                cv2.rectangle(frame, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (255, 255, 0), 2)
            for item in finalArray:
                cv2.rectangle(frame, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (255, 0, 255), 2)
            print(rectArray)
            cv2.namedWindow("enhanced", 0)
            cv2.resizeWindow("enhanced", 960, 540)
            cv2.imshow("enhanced", frame)
            cv2.waitKey(0)

    end_num = -1

    while end_num + 1 < len(data):
        current_second = data[end_num + 1][0]
        frame_array = np.where(data[:, 0] == current_second)
        end_num = frame_array[0][len(frame_array[0]) - 1]
        if math.floor(float(current_second) * framerate) == 0:
            print('current_second:', current_second, ' Skip due to a black screen')
            continue
        cutpic(data[frame_array[0]])

def filterLastImagePerId(videoFolder, camera):
    imgFolder = videoFolder + '/images/camera' + str(camera)+'/'
    cropPath = videoFolder + '/images/cropped/'

    if not os.path.exists(cropPath):
        os.makedirs(cropPath)

    files = os.listdir(imgFolder)
    record = open(videoFolder + '/images/rand_cutinfo.txt', 'a+')

    crop_rate = 0.3
    side_rate = 0.3

    arr = []
    frame = []
    name = []
    startframe = []
    startname = []

    for file in files:
        if '.jpg' in file:
            # delete the final images of each ID
            id = int(file.split('_')[0])
            frame_num = file.split('_')[3].replace('.jpg','').replace('f','')
            frame_num = int(frame_num)
            if id not in arr:
                arr.append(id)
                frame.append(frame_num)
                name.append(file)
                startframe.append(frame_num)
                startname.append(file)
            else:
                id_index = arr.index(id)
                if frame_num > frame[id_index]:
                    frame[id_index] = frame_num
                    name[id_index] = file
                if frame_num < startframe[id_index]:
                    startframe[id_index] = frame_num
                    startname[id_index] = file

            # random crop
            img = Image.open(imgFolder + file)
            W = img.width
            H = img.height
            random_index = random.randint(0, 2)
            x1 = random.randint(0, int(side_rate * W))
            x2 = W - random.randint(0, int(side_rate * W))
            if random_index == 0:
                x2 = W
            elif random_index == 1:
                x1 = 0
            randomnum = random.randint(0, 100) / 100
            if randomnum > crop_rate:
                cropped = img.crop((x1, 0, x2, H))
                record.write(file+','+str(x1)+',0,'+str(x2)+','+str(H)+'\n')
                cropped.save(cropPath + file)
            else:
                y2 = H - random.randint(0, int(0.5 * H))
                y1 = random.randint(0, int(0.1 * H))
                cropped = img.crop((x1, y1, x2, y2))
                record.write(file+','+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+'\n')
                cropped.save(cropPath + file)
    for item in name:
        os.remove(imgFolder + item)

for i in range(0, len(processList)):
    try:
        scene, camera, dealTime = processList[i]
        videoFolder = 'Data/Recordings' + str(dealTime) + '_scene' + str(scene).zfill(2)
        extractLabel(videoFolder, camera)
        cropDataset(videoFolder, scene, camera)
        filterLastImagePerId(videoFolder, camera)
        print('Finished:', videoFolder, camera)
    except:
        print('wrong:',scene, camera, dealTime)
