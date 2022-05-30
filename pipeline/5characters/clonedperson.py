import argparse
import os.path as osp
import torch
from torch.backends import cudnn
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw
import cv2
from reid.models import resmap_texture
from reid.models.qaconv import QAConv
from reid.utils.data import transforms as T
from reid.utils.serialization import load_checkpoint
from reid.loss.pairwise_matching_loss import PairwiseMatchingLoss
from reid.evaluators import extract_features, pairwise_distance
from reid.utils.data.preprocessor import Preprocessor, Preprocessor_gen
import tensorflow as tf
import os
import random
import time
import csv
import shutil

def main(args):
    # read csv file
    def readCSV2List(readCSV2ListFilePath):
        csv_list = csv.reader(open(readCSV2ListFilePath))
        result = []
        for line in csv_list:
            result.append(line)
        return np.array(result)

    def mergePic(img1, img2):
        try:
            L, H = img1.size
            color_0 = (0, 0, 0)
            for h in range(H):
                for l in range(L):
                    dot = (l, h)
                    color_1 = img1.getpixel(dot)
                    if color_1 == color_0 or color_1 == (0, 0, 0, 0):
                        img1.putpixel(dot, img2.getpixel(dot))
            return img1
        except:
            return img1

    name_prefix = args.name_prefix
    result_folder = args.result + '/'
    result_models = result_folder + 'models/'
    result_clothes = result_folder + 'clothes/'
    result_textures = result_folder + 'textures/'
    regular_path, irregular_path, cell_path, step_path = result_textures + '/regular/', result_textures + '/irregular/', result_textures + '/cell/', result_textures + '/step/'
    handle_type = args.type

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        os.makedirs(result_models)
        os.makedirs(result_clothes)
        os.makedirs(result_textures)
        os.makedirs(regular_path)
        os.makedirs(irregular_path)
        os.makedirs(cell_path)
        os.makedirs(step_path)

    map_record = open(args.result+"/correspondence_table_of_images_and_characters.csv", 'a')
    map_record.write('character,image,gender\n')

    # settings
    person_ID = args.start_id
    resource_folder = args.resources+'/'
    source_folder = args.source+'/'
    save_step_open = args.show_step
    print('save_step_open:', save_step_open)
    step = 2
    if handle_type == 'dress':
        step = 1

    model_list_allset = readCSV2List(resource_folder + 'model.csv')
    random.shuffle(model_list_allset)
    clothList = readCSV2List(source_folder + '/' + args.landmarks)


    def PerspectiveHomography(clothArry1, modelArray, clothArry2, clothIndex):

        # get clothes keypoints position, since index are not in a normal order
        def getLabels(getLabels_index, getLabels_list):
            # insert estimable value
            if len(getLabels_list) == 38:
                getLabels_list.append(str(int((int(float(getLabels_list[22])) + int(float(getLabels_list[24]))) / 2)))
                getLabels_list.append(str(int((int(float(getLabels_list[23])) + int(float(getLabels_list[25]) / 2)))))
            if len(getLabels_list) == 14:
                getLabels_list.append(str(int((int(float(getLabels_list[0])) + int(float(getLabels_list[2]) / 2)))))
                getLabels_list.append(str(int((int(float(getLabels_list[1])) + int(float(getLabels_list[3]) / 2)))))
            if len(getLabels_list) == 44:
                getLabels_list.append(str(int((int(float(getLabels_list[18])) + int(float(getLabels_list[34]) / 2)))))
                getLabels_list.append(str(int((int(float(getLabels_list[19])) + int(float(getLabels_list[35]) / 2)))))
            getLabels_result = []
            for getLabels_i in range(len(getLabels_index)):
                try:
                    getLabels_result.append([int(float(getLabels_list[getLabels_index[getLabels_i] * 2 - 2].replace('"', ''))),int(float(getLabels_list[getLabels_index[getLabels_i] * 2 - 1].replace('"', '')))])
                except:
                    print('Error: getLabels wrong,', getLabels_i, getLabels_index, getLabels_list)
            return np.array(getLabels_result)

        def addMask(img, modelPoint):
            img = img.convert('RGB')
            L, H = img.size
            for h in range(H):
                for l in range(L):
                    dot = (l, h)
                    if cv2.pointPolygonTest(modelPoint, dot, False) < 0:
                        img.putpixel(dot, (0, 0, 0))
            return img

        def getpoints(point_str):
            if point_str == "":
                return []
            arr1 = point_str.split('-')
            for i in range(len(arr1)):
                arr1[i] = arr1[i].split(',')
                for j in range(len(arr1[i])):
                    arr1[i][j] = list(map(int, arr1[i][j].split(' ')))
            return arr1

        cloth_img = Image.open(clothArry1[0]).convert('RGB')
        cloth_keypoints_order = getpoints(modelArray[3])

        model_name = modelArray[0]

        model_img_path, model_landmark = 'images/' + modelArray[0], modelArray[2].split(',')

        # model_img = cv2.imread(resource_folder + model_img_path)
        model_img = cv2.cvtColor(np.zeros((1000,1000), dtype=np.uint8), cv2.COLOR_GRAY2RGB)

        model_keypoints_order = getpoints(modelArray[4])
        save_name = model_name.split('.')[0] + "_" + str(clothIndex) + '_'

        # registered clothes mapping
        homography_result = np.zeros((model_img.shape[0], model_img.shape[1], 3), np.uint8)
        clothKeypoints = clothArry1[1]
        ph_model_index = 0
        for i in range(0, len(cloth_keypoints_order)):
            for j in range(0, len(cloth_keypoints_order[i])):
                cloth_keypoints_position = getLabels(cloth_keypoints_order[i][j], clothKeypoints)

                model_keypoints_position = getLabels(model_keypoints_order[0][ph_model_index], model_landmark)
                ph_model_index += 1
                cloth_area_img = cv2.cvtColor(np.array(addMask(cloth_img, cloth_keypoints_position)), cv2.COLOR_BGR2RGB)


                H, mask = cv2.findHomography(cloth_keypoints_position, model_keypoints_position)
                img_warp = cv2.warpPerspective(cloth_area_img, H, (model_img.shape[1], model_img.shape[0]))
                img_warp = addMask(Image.fromarray(cv2.cvtColor(img_warp, cv2.COLOR_BGR2RGB)), model_keypoints_position)

                if i == 0 and j == 0:
                    homography_result = img_warp
                else:
                    homography_result = mergePic(img_warp, homography_result)
                if save_step_open:
                    cv2.imwrite(step_path + save_name + 'cloth_area_img' + str(i) + '.jpg', cloth_area_img)
                    homography_result.save(step_path + save_name + 'homography_result' + str(i) + '.jpg')
            cloth_img = Image.open(clothArry2[0]).convert('RGB')
            clothKeypoints = clothArry2[1]
        return homography_result, cloth_img, model_landmark, save_name, model_img

    def HomoGeneousExpansion(modelLandmark, clothLandmark, clothTypeIndex, clothImg, sourceFolder, clothPath, model, save_step_open, modelImg, bgscale):

        # If pattern in the clothes area
        def getBinaryTensor(myTensor, boundary):
            one = tf.ones_like(myTensor)
            zero = tf.zeros_like(myTensor)
            return tf.where(myTensor >= boundary, one, zero)

        # get clothe cell
        def getMaxArea(getMaxAreaImg, getMaxAreaScoreList, featureH, featureW):
            getMaxAreaImg = getMaxAreaImg.copy()
            L, H = getMaxAreaImg.size
            [std, row, col, ker_size] = getMaxAreaScoreList[0]
            if L / H < 128 / 384:
                per_size = H / featureH
            else:
                per_size = L / featureW
            x1 = int(col * per_size)
            y1 = int(row * per_size)
            size = int(ker_size * per_size)
            if x1 + size > L:
                size = L - x1
            if y1 + size > H:
                size = H - y1
            x2 = x1 + size
            y2 = y1 + size
            cloth_cell = Image.fromarray(np.array(getMaxAreaImg)[y1:y2, x1:x2])
            draw = ImageDraw.ImageDraw(getMaxAreaImg)
            draw.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red', width=2)
            return cloth_cell, [x1, y1, x2, y2], getMaxAreaImg

        def expandBackground(clothCell):
            cell_w, cell_h = clothCell.size
            expand_background_img = Image.new("RGB", (1000, 1000), (0, 0, 0))
            for h in range(0, 1000):
                for l in range(0, 1000):
                    row = int(h / cell_h)
                    col = int(l / cell_w)
                    dot = (l, h)
                    if row % 2 == 0 and col % 2 == 0:
                        color_1 = clothCell.getpixel((l % cell_w, h % cell_h))
                        expand_background_img.putpixel(dot, (color_1[0], color_1[1], color_1[2], 255))
                    elif row % 2 == 1 and col % 2 == 0:
                        color_1 = clothCell.getpixel((l % cell_w, cell_h - 1 - h % cell_h))
                        expand_background_img.putpixel(dot, (color_1[0], color_1[1], color_1[2], 255))
                    elif row % 2 == 1 and col % 2 == 1:
                        color_1 = clothCell.getpixel((cell_w - 1 - l % cell_w, cell_h - 1 - h % cell_h))
                        expand_background_img.putpixel(dot, (color_1[0], color_1[1], color_1[2], 255))
                    else:
                        color_1 = clothCell.getpixel((cell_w - 1 - l % cell_w, h % cell_h))
                        expand_background_img.putpixel(dot, (color_1[0], color_1[1], color_1[2], 255))
            return expand_background_img

        with torch.no_grad():
            # points of real images
            mask_point = ','.join(clothLandmark).replace('"', '').split(',')
            mask_point = np.array(list(map(int, mask_point))).reshape(-1, 2)
            if clothTypeIndex == 4 and len(mask_point)>2 and mask_point[6][1] - mask_point[8][1] >= mask_point[2][0] - mask_point[1][0]:
                mask_point[0][1] = mask_point[8][1]
                mask_point[1][1] = mask_point[8][1]
                mask_point[2][1] = mask_point[8][1]
                mask_point[3][1] = mask_point[8][1]
                mask_point[13][1] = mask_point[8][1]

            line1 = mask_point[:, 0]
            line2 = mask_point[:, 1]
            x1 = np.min(line1)
            y1 = np.min(line2)
            x2 = np.max(line1)
            y2 = np.max(line2)
            clothes_area_keypoints = np.zeros(mask_point.shape)
            clothes_area_keypoints[:, 0] = mask_point[:, 0] - x1
            clothes_area_keypoints[:, 1] = mask_point[:, 1] - y1

            # -1 when the keypoint is not detected
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            clothes_area_img = Image.fromarray(np.array(clothImg)[y1:y2, x1:x2])

            L, H = clothes_area_img.size
            if L / H < 128 / 384:
                scale = 384 / H
                clothes_area_resize_img = clothes_area_img.resize((int(L * scale), 384), Image.LANCZOS)
            else:
                scale = 128 / L
                clothes_area_resize_img = clothes_area_img.resize((128, int(scale * H)), Image.LANCZOS)

            clothes_area_keypoints = (clothes_area_keypoints * scale).astype(int)

            filter_mask = Image.new("RGBA", clothes_area_resize_img.size, (0, 0, 0, 0))
            filter_mask_operate = ImageDraw.Draw(filter_mask)
            filter_mask_operate.polygon(np.float32(clothes_area_keypoints), fill='#ffffff')
            extract_feature_img = Image.new("RGB", (128, 384), (0, 0, 0))
            extract_feature_img.paste(clothes_area_resize_img, (0, 0), filter_mask)

            cloth_mask_img = Image.new("RGB", (128, 384), (0, 0, 0))
            cloth_mask_operate = ImageDraw.Draw(cloth_mask_img)
            cloth_mask_operate.polygon(np.float32(clothes_area_keypoints), fill='#ffffff')

            data = []
            data.append((extract_feature_img, sourceFolder + clothPath, 0, 0))

            data_loader = DataLoader(Preprocessor_gen(data, transform=transformer), batch_size=64, num_workers=8,
                                     shuffle=False, pin_memory=True)
            features, _ = extract_features(model, data_loader)
            features = {key: features[key].cuda() for key in features}

            sigma_list = []
            # get cloth segmentation
            for index, feature in features.items():
                img_feature = feature.unsqueeze(0)
                # C: 512, feature_H: 48, feature_W: 16
                img_num, C, feature_H, feature_W = img_feature.shape
                # print('C:', img_num, C, '  feature_H:', feature_H, '  feature_W:', feature_W)
                mask_resize = cloth_mask_img.resize((feature_W, feature_H), Image.NEAREST)

                mask_point = np.array(mask_resize.convert('1')) + 0.0
                mask_point = torch.from_numpy(mask_point)
                mask_point = mask_point.view(1, 1, feature_H, feature_W)

                for ker_size in range(2, int(feature_W / 2) + 1):
                    unfold = nn.Unfold(ker_size, padding=0)
                    kernel_mask = unfold(mask_point)
                    mask_list = kernel_mask.permute([0, 2, 1])
                    area_list = mask_list.sum(dim=2)
                    eff_area_list = getBinaryTensor(area_list, ker_size * ker_size * 1).numpy()

                    kernel = unfold(img_feature)
                    B, C_kh_kw, L = kernel.size()
                    kernel = kernel.view(B, C, ker_size * ker_size, L)
                    std_list = F.normalize(kernel, dim=2).std(dim=2)
                    std_list = std_list.permute([0, 2, 1])
                    mean_list = std_list.mean(dim=2) / (ker_size * ker_size)
                    mean_index = 0

                    for row in range(0, feature_H + 1 - ker_size):
                        for col in range(0, feature_W + 1 - ker_size):
                            if eff_area_list[0][mean_index] > 0:
                                sigma_list.insert(0, [mean_list[0][mean_index], row, col, ker_size])
                            mean_index += 1

            sigma_list = np.array(sigma_list)

            final_score_arr = sigma_list[sigma_list[:, 0].argsort()]

            clothes_pattern_img, area_points, cloth_withdraw_img = getMaxArea(clothes_area_img, final_score_arr, feature_H,feature_W)
            cpw, cph = clothes_pattern_img.size
            # points of clothes models

            homo_point = ','.join(modelLandmark).replace('"', '').split(',')
            if len(homo_point) > 1:
                homo_point = np.array(list(map(int, homo_point))).reshape(-1, 2)

                front1 = homo_point[:, 0]
                front2 = homo_point[:, 1]
                front_area = [np.min(front1), np.min(front2), np.max(front1), np.max(front2)]
            else:
                front_area = [int(int(bgscale[0])*cpw/50), int(int(bgscale[1])*cph/50),int(int(bgscale[2])*cpw/50), int(int(bgscale[3])*cph/50)]

            cloth_W, cloth_H = clothes_area_img.size
            area_W, area_H = area_points[2] - area_points[0], area_points[3] - area_points[1]
            texture_W, texture_H = front_area[2] - front_area[0], front_area[3] - front_area[1]
            scale_W, scale_H = max(int(area_W * texture_W / cloth_W), 1), max(int(area_H * texture_H / cloth_H), 1)

            scaled_cloth_cell_img = clothes_pattern_img.resize((scale_W, scale_H), Image.LANCZOS)
            expand_irregular_background_img_origin = expandBackground(clothes_pattern_img)
            clothes_pattern_img = clothes_pattern_img.resize((int(area_W * (int(bgscale[2])-int(bgscale[0])) / 50), int(area_H * (int(bgscale[3])-int(bgscale[1])) / 50)), Image.LANCZOS)
            expand_regular_background_img = expandBackground(scaled_cloth_cell_img)
            expand_irregular_background_img = expandBackground(clothes_pattern_img)

            if save_step_open:
                clothes_area_img.save(step_path + save_name + 'clothes_area_img.png')
                extract_feature_img.save(step_path + save_name + 'extract_feature_img.png')
                cloth_mask_img.save(step_path + save_name + 'cloth_mask_img.png')
                cloth_withdraw_img.save(step_path + save_name + 'cloth_withdraw_img.png')
                scaled_cloth_cell_img.save(step_path + save_name + 'scaled_cloth_cell_img.png')
                expand_regular_background_img.save(step_path + save_name + 'expand_regular_background_img.png')

                expand_regular_background_img.putpixel((scale_W, scale_H), (255, 255, 255, 255))
                expand_regular_background_img.putpixel((scale_W - 1, scale_H), (255, 255, 255, 255))
                expand_regular_background_img.putpixel((scale_W + 1, scale_H), (255, 255, 255, 255))
                expand_regular_background_img.putpixel((scale_W, scale_H - 1), (255, 255, 255, 255))
                expand_regular_background_img.putpixel((scale_W, scale_H + 1), (255, 255, 255, 255))
                expand_regular_background_img.putpixel((scale_W - 1, scale_H - 1), (255, 255, 255, 255))
                expand_regular_background_img.save(step_path + save_name + 'expand_regular_background_img_marked.png')

                expand_irregular_background_img.putpixel((scale_W, scale_H), (255, 255, 255, 255))
                expand_irregular_background_img.putpixel((scale_W - 1, scale_H), (255, 255, 255, 255))
                expand_irregular_background_img.putpixel((scale_W + 1, scale_H), (255, 255, 255, 255))
                expand_irregular_background_img.putpixel((scale_W, scale_H - 1), (255, 255, 255, 255))
                expand_irregular_background_img.putpixel((scale_W, scale_H + 1), (255, 255, 255, 255))
                expand_irregular_background_img.putpixel((scale_W - 1, scale_H - 1), (255, 255, 255, 255))
                expand_irregular_background_img.save(
                    step_path + save_name + 'expand_irregular_background_img_marked.png')

                resize_cloth_cell_img = clothes_pattern_img.resize((modelImg.shape[0], modelImg.shape[1]),
                                                                   Image.LANCZOS)
                resize_cloth_cell_img.save(step_path + save_name + 'resize_cloth_cell_img.png')
                resize_regular_uvmap_img = mergePic(homography_result.copy(), resize_cloth_cell_img)
                resize_regular_uvmap_img.save(step_path + save_name + 'resize_regular_uvmap_img.png')
            return expand_regular_background_img, expand_irregular_background_img, clothes_pattern_img, final_score_arr, area_points, expand_irregular_background_img_origin

    def create_pair_person(clist, personIndex):
        def createClothes(modelType, uuid, saveName, resultTextures, imgName, imgPath):
            savePath = resultTextures + '/' + str(modelType) + '/'
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            for filename in os.listdir('models/' + modelType + '/'):
                if 'mhclo' not in filename and 'mhmat' not in filename:
                    shutil.copy('models/' + modelType + '/' + filename, savePath)
                else:
                    if 'mhclo' in filename:
                        with open('models/' + str(modelType) + '/' + filename, "r", encoding="utf-8") as f1, open(savePath + '' + saveName + ".mhclo", "a", encoding="utf-8") as f2:
                            for line in f1:
                                if 'uuid' in line:
                                    line = 'uuid ' + uuid + '\n'
                                if 'name' in line:
                                    line = 'name ' + str(saveName) + '\n'
                                if 'material' in line:
                                    line = 'material ' + str(saveName) + '.mhmat' + '\n'
                                f2.write(line)
                    else:
                        with open('models/' + str(modelType) + '/' + filename, "r", encoding="utf-8") as f1, open(
                                savePath + saveName + ".mhmat", "a", encoding="utf-8") as f2:
                            for line in f1:
                                if 'name' in line:
                                    line = 'name ' + str(saveName) + '\n'
                                if 'diffuseTexture' in line:
                                    line = 'diffuseTexture ' + imgName + '\n'
                                f2.write(line)
                            shutil.move(imgPath, savePath + imgName)

        def generateHuman(clothList, personId, gender):
            # load acc
            nanfa = open('accessories/hair_men.txt', 'r').readlines()
            nvfa = open('accessories/hair_wemen.txt', 'r').readlines()
            xie = open('accessories/shoes.txt', 'r').readlines()
            pifu = open('accessories/skin.txt', 'r').readlines()
            huxu = open('accessories/beard.txt', 'r').readlines()

            if gender > 0:
                Gender1 = 1000000
            else:
                Gender1 = 0
            #     setting
            Gender = '%.6f' % (Gender1 / 1000000)
            Muscle = '%.6f' % (random.randint(0, 1000000) / 1000000)
            African_1 = random.randint(0, 1000000)
            African = '%.6f' % (African_1 / 1000000)
            Asian_1 = random.randint(0, 1000000 - African_1)
            Asian = '%.6f' % (Asian_1 / 1000000)
            Caucasian = '%.6f' % ((1000000 - Asian_1 - African_1) / 1000000)
            if Gender1 > 1000000 / 2:
                m_height = random.gauss(170, 5.7) / 200
                while m_height > 1:
                    m_height = random.gauss(170, 5.7) / 200
                Height = '%.6f' % (m_height)
            else:
                m_height = random.gauss(160, 5.2) / 200
                while m_height > 1:
                    m_height = random.gauss(160, 5.2) / 200
                Height = '%.6f' % (m_height)
            BreastSize = '%.6f' % (random.randint(0, 70) / 100)
            Age = '%.6f' % (random.randint(20, 90) / 100)
            BreastFirmness = '%.6f' % (random.randint(30, 100) / 100)
            Weight = '%.6f' % (random.randint(0, 1000000) / 1000000)

            file_name = name_prefix + str(personId)
            # creating person file
            f = open(result_models + file_name + ".mhm", 'a')
            f.write('# Written by MakeHuman 1.1.1\n')
            f.write('version v1.1.1\n')
            f.write('tags ' + file_name + '\n')
            f.write('camera 0.0 0.0 0.0 0.0 0.0 1.0\n')
            f.write('modifier macrodetails-universal/Muscle ' + Muscle + '\n')
            f.write('modifier macrodetails/African ' + African + '\n')
            f.write('modifier macrodetails-proportions/BodyProportions 0.500000\n')
            f.write('modifier macrodetails/Gender ' + Gender + '\n')
            f.write('modifier macrodetails-height/Height ' + Height + '\n')
            f.write('modifier breast/BreastSize ' + BreastSize + '\n')
            f.write('modifier macrodetails/Age ' + Age + '\n')
            f.write('modifier breast/BreastFirmness ' + BreastFirmness + '\n')
            f.write('modifier macrodetails/Asian ' + Asian + '\n')
            f.write('modifier macrodetails/Caucasian ' + Caucasian + '\n')
            f.write('modifier macrodetails-universal/Weight ' + Weight + '\n')
            f.write('skeleton cmu_mb.mhskel\n')
            f.write('eyes HighPolyEyes 2c12f43b-1303-432c-b7ce-d78346baf2e6\n')

            # adding clothes
            if Gender1 > 1000000 / 2:
                # 70% man adding beard
                if random.randint(0, 9) > 7:
                    f.write(huxu[random.randint(0, len(huxu) - 1)])
                f.write(nanfa[random.randint(0, len(nanfa) - 1)])
            else:
                f.write(nvfa[random.randint(0, len(nvfa) - 1)])

            f.write(xie[random.randint(0, len(xie) - 1)])
            for i in range(0, len(clothList)):
                f.write(clothList[i] + '\n')
            f.write('clothesHideFaces True\n')
            f.write(pifu[random.randint(0, len(pifu) - 1)])
            f.write('material Braid01 eead6f99-d6c6-4f6b-b6c2-210459d7a62e braid01.mhmat\n')
            f.write('material HighPolyEyes 2c12f43b-1303-432c-b7ce-d78346baf2e6 eyes/materials/brown.mhmat\n')
            f.write('subdivide False\n')
            f.close()

        gender = personIndex % 2
        pair_list = []
        for i in range(0, len(clist)):
            modelType = clist[i][0].split('.')[0]
            if (int(modelType) > 88 and int(modelType) < 140) or (int(modelType) > 154 and int(modelType) < 160) or int(modelType) > 167:
                gender = 0
            uuid = '20220218-0000-0000-' + str(int(modelType)).zfill(4) + '-' + str(personIndex).zfill(12)
            save_name = handle_type + '_' + modelType + '_' + str(personIndex).zfill(7)
            save_folder = result_clothes

            img_path_list = clist[i][1].split('/')
            img_name = img_path_list[len(img_path_list) - 1]
            pair_list.append('clothes ' + save_name + ' ' + uuid)
            img_path = clist[i][1]
            createClothes(modelType, uuid, save_name, save_folder, img_name, img_path)

        generateHuman(pair_list, personIndex, gender)
        map_record.write(name_prefix+str(personIndex) + ',' + clist[0][2] + ',' + str(gender) + '\n')

    cudnn.deterministic = False
    cudnn.benchmark = True

    # Create model
    model = resmap_texture.create(args.arch, ibn_type=args.ibn, final_layer=args.final_layer, neck=args.neck).cuda()
    num_features = model.num_features
    feamap_factor = {'layer2': 8, 'layer3': 16, 'layer4': 32}
    hei = args.height // feamap_factor[args.final_layer]
    wid = args.width // feamap_factor[args.final_layer]
    matcher = QAConv(num_features, hei, wid).cuda()

    transformer = T.Compose([
        T.Resize((args.height, args.width), interpolation=3),
        T.ToTensor(),
    ])

    # Criterion
    criterion = PairwiseMatchingLoss(matcher).cuda()

    # Load from checkpoint
    print('Loading checkpoint...')

    checkpoint = load_checkpoint('checkpoint.pth.tar')
    model.load_state_dict(checkpoint['model'])
    criterion.load_state_dict(checkpoint['criterion'])

    model = nn.DataParallel(model).cuda()

    # clothes types (0-17) and corresponding textures
    style_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '8': 5, '13': 6, '17': 7, '14': 8, '15': 9, '18': 10, '24': 11, '25': 12, '28': 13, '34': 14, '35': 15, '38': 16}
    model_array = [0]*17
    for cloth_index in range(0, len(clothList), step):
        if not (len(clothList[cloth_index]) > 0 and os.path.exists(source_folder + clothList[cloth_index][0])):
            continue
        cloth_path, cloth_type_index, cloth_landmark = clothList[cloth_index][0], int(clothList[cloth_index][1]), clothList[cloth_index][2].split(',')
        cloth_path2, cloth_type_index2, cloth_landmark2 = clothList[cloth_index+step-1][0], int(clothList[cloth_index+step-1][1]), clothList[cloth_index+step-1][2].split(',')

        print(cloth_index, cloth_path, cloth_type_index)
        cloth_type_index3 = str(cloth_type_index) + str(cloth_type_index2)
        if cloth_type_index > cloth_type_index2:
            cloth_type_index3 = str(cloth_type_index2) + str(cloth_type_index)

        model_list = model_list_allset[np.where(model_list_allset[:,1] == str(cloth_type_index))]
        model_list2 = model_list_allset[np.where(model_list_allset[:, 1] == str(cloth_type_index2))]
        model_list3 = model_list_allset[np.where(model_list_allset[:, 1] == cloth_type_index3)]

        choose_type = random.randint(0, len(model_list3) + min(len(model_list), len(model_list2)))

        imgUrl_list = []
        if choose_type >= len(model_list3):
            # Use separate models
            model_index = model_array[style_dict[str(cloth_type_index)]]
            model_array[style_dict[str(cloth_type_index)]] += 1

            homography_result, cloth_img, model_landmark, save_name, model_img = PerspectiveHomography([source_folder + cloth_path, cloth_landmark], model_list[model_index%len(model_list)], [source_folder + cloth_path2, cloth_landmark2], cloth_index)
            scale_array = model_list[model_index%len(model_list)][5].split(',')
            expand_regular_background_img, expand_irregular_background_img, clothes_pattern_img, final_score_arr, area_points, expand_irregular_background_img_origin = HomoGeneousExpansion(model_landmark, cloth_landmark, cloth_type_index,cloth_img, source_folder, cloth_path, model, save_step_open, model_img, scale_array)

            if len(model_list[model_index%len(model_list)][2].split(',')) > 1:
                regular_uvmap_img = mergePic(homography_result.copy(), expand_regular_background_img)
            else:
                regular_uvmap_img = expand_regular_background_img

            regular_uvmap_img.save(regular_path + save_name + 'regular.png')
            expand_irregular_background_img.save(irregular_path + save_name + 'irregular.jpeg')
            clothes_pattern_img.save(cell_path + save_name + str(round(final_score_arr[0][0].item(), 6)) +'_'+ str(area_points[0]) + '_' + str(area_points[1]) + '_' + str(area_points[2]) + '_' + str(area_points[3]) + '_cell.jpeg', "JPEG")

            imgUrl_list.append([model_list[model_index%len(model_list)][0],regular_path + save_name + 'regular.png', source_folder + cloth_path])
            if step > 1:
                model_index2 = model_array[style_dict[str(cloth_type_index2)]]
                model_array[style_dict[str(cloth_type_index2)]] += 1

                homography_result2, cloth_img2, model_landmark2, save_name2, model_img2 = PerspectiveHomography([source_folder + cloth_path2, cloth_landmark2], model_list2[model_index2%len(model_list2)],[source_folder + cloth_path, cloth_landmark], cloth_index+1)
                scale_array2 = model_list2[model_index2%len(model_list2)][5].split(',')

                expand_regular_background_img2, expand_irregular_background_img2, clothes_pattern_img2, final_score_arr2, area_points2, expand_irregular_background_img_origin2 = HomoGeneousExpansion(model_landmark2, cloth_landmark2, cloth_type_index2, cloth_img2, source_folder, cloth_path2, model,save_step_open, model_img2, scale_array2)

                if len(model_list2[model_index2%len(model_list2)][2].split(',')) > 1:
                    regular_uvmap_img2 = mergePic(homography_result2.copy(), expand_regular_background_img2)
                else:
                    regular_uvmap_img2 = expand_regular_background_img2

                regular_uvmap_img2.save(regular_path + save_name2 + 'regular.png')
                expand_irregular_background_img2.save(irregular_path + save_name2 + 'irregular.jpeg')
                clothes_pattern_img2.save(cell_path + save_name2 + str(round(final_score_arr2[0][0].item(), 6)) + '_' + str(area_points2[0]) + '_' + str(area_points2[1]) + '_' + str(area_points2[2]) + '_' + str(area_points2[3]) + '_cell.jpeg', "JPEG")

                imgUrl_list.append([model_list2[model_index2%len(model_list2)][0], regular_path + save_name2 + 'regular.png', source_folder + cloth_path])
        else:
            # Use combined model
            model_index = model_array[style_dict[str(cloth_type_index3)]]
            model_array[style_dict[str(cloth_type_index3)]] += 1

            if cloth_type_index > cloth_type_index2:
                a, b, c = cloth_path, cloth_type_index, cloth_landmark
                cloth_path, cloth_type_index, cloth_landmark = cloth_path2, cloth_type_index2, cloth_landmark2
                cloth_path2, cloth_type_index2, cloth_landmark2 = a, b, c
            homography_result, cloth_img, model_landmark, save_name, model_img = PerspectiveHomography([source_folder + cloth_path, cloth_landmark], model_list3[model_index%len(model_list3)],[source_folder + cloth_path2, cloth_landmark2], cloth_index)

            scale_array3 = model_list3[model_index%len(model_list3)][5].split(',')

            expand_regular_background_img, expand_irregular_background_img, clothes_pattern_img, final_score_arr, area_points, expand_irregular_background_img_origin = HomoGeneousExpansion(model_landmark, cloth_landmark, cloth_type_index, cloth_img, source_folder, cloth_path, model, save_step_open, model_img, scale_array3)
            expand_regular_background_img2, expand_irregular_background_img2, clothes_pattern_img2, final_score_arr2, area_points2, expand_irregular_background_img_origin = HomoGeneousExpansion(model_landmark, cloth_landmark2, cloth_type_index2, cloth_img, source_folder, cloth_path2, model, save_step_open, model_img, scale_array3)
            replace_area = model_list3[model_index%len(model_list3)][6].split(',')

            for i in range(len(replace_area)):
                pants_points = replace_area[i].split(' ')
                if len(pants_points) > 1:
                    x1, y1, x2, y2 = int(pants_points[0]), int(pants_points[1]), int(pants_points[2]), int(pants_points[3])
                    box = expand_regular_background_img2.crop((x1,y1,x2,y2))
                    expand_regular_background_img.paste(box,(x1, y1))
            if len(model_list3[model_index%len(model_list3)][2].split(',')) > 1:
                regular_uvmap_img = mergePic(homography_result.copy(), expand_regular_background_img)
            else:
                regular_uvmap_img = expand_regular_background_img
            regular_uvmap_img.save(regular_path + save_name + 'regular.png')
            expand_irregular_background_img.save(irregular_path + save_name + 'irregular.jpeg')
            clothes_pattern_img.save(cell_path + save_name + str(round(final_score_arr[0][0].item(), 6)) + '_' + str(area_points[0]) + '_' + str(area_points[1]) + '_' + str(area_points[2]) + '_' + str(area_points[3]) + '_cell.jpeg', "JPEG")

            imgUrl_list.append([model_list3[model_index%len(model_list3)][0],regular_path + save_name + 'regular.png', source_folder + cloth_path])
        create_pair_person(imgUrl_list, person_ID)
        person_ID += 1

if __name__ == '__main__':
    my_start = time.time()

    parser = argparse.ArgumentParser(description="QAConv")
    # data
    parser.add_argument('--height', type=int, default=384, help="height of the input image, default: 384")
    parser.add_argument('--width', type=int, default=128, help="width of the input image, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=resmap_texture.names(), help="the backbone network, default: resnet50")
    parser.add_argument('--final_layer', type=str, default='layer3', choices=['layer2', 'layer3', 'layer4'], help="the final layer, default: layer3")
    parser.add_argument('--neck', type=int, default=64, help="number of channels for the final neck layer, default: 64")
    parser.add_argument('--ibn', type=str, choices={'a', 'b'}, default='b', help="IBN type. Choose from 'a' or 'b'. Default: None")
    # setting
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--result', type=str, metavar='PATH', default=osp.join(working_dir, 'Result'), help="the path of the generated characters")
    parser.add_argument('--source', type=str, metavar='PATH', default=osp.join(working_dir, 'Source'), help="the path of images")
    parser.add_argument('--resources', type=str, metavar='PATH', default=osp.join(working_dir, 'Resources/'), help="the path of MH community clothes models")
    parser.add_argument('--start_id', type=int, default=0, help="Character ID starting value")
    parser.add_argument('--type', type=str, default='pair', help="Cloth type: dress/pair")
    parser.add_argument('--name_prefix', type=str, default='B', help="Prefix of generated characters")
    parser.add_argument('--landmarks', type=str, default='test.csv', help="CSV file of clothes landmarks")
    parser.add_argument('--show_step', action="store_true", help="Save the process map or not")

    main(parser.parse_args())
    my_end=time.time()

    print('running time: ',my_end-my_start)
