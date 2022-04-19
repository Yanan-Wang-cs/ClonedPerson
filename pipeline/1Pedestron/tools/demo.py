import argparse

import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import time
import cv2
import torch
import glob
import json
import mmcv
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result
import shutil

my_start = time.time()
def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_img_dir', type=str, help='the dir of input images')
    parser.add_argument('output_dir', type=str, help='the dir for result images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')
    args = parser.parse_args()
    return args



def mock_detector(model, image_name, output_dir, input_dir):
    image = cv2.imread(image_name)
    results = inference_detector(model, image)
    basename = os.path.basename(image_name).split('.')[0]
    result_name = basename + "_result.jpg"
    result_name = os.path.join(output_dir, result_name)
    boxes = show_result(image, results, model.CLASSES, out_file=result_name)
    img = Image.open(image_name).convert('RGB')
    W,H = img.size
    try:
        if boxes[0][4]>= 0.8:
            x1 = int(boxes[0][0])
            y1 = int(boxes[0][1])
            x2 = int(boxes[0][2])
            y2 = int(boxes[0][3])

        ######## only keep persons whose size is big enough ############
            if (y2-y1)*(x2-x1)/(W*H)>=0.2:
                img = np.array(img)
                img2 = img[y1:y2, x1:x2]
                img2 = Image.fromarray(img2)
                img2.save(output_dir+'{:.6f}'.format(boxes[0][4])+'_'+str(x1)+'_'+str(y1)+'_'+ str(x2)+'_'+ str(y2)+'_'+image_name.replace(input_dir,''))

            ##################draw detect boxes ##################

                #img = Image.fromarray(img)
                #a = ImageDraw.ImageDraw(img)
                #a.rectangle(((x1,y1),(x2,y2)),fill=None,outline='red',width=3)
                #a.text((x1,y1-10),'{:.6f}'.format(boxes[0][4]),fill=(255,0,0))
                #img.save(output_dir+'{:.6f}'.format(boxes[0][4])+'_result_'+image_name.replace(input_dir,''))
    except:
        print('no detection:', image_name)

def create_base_dir(dest):
    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

def run_detector_on_dataset():
    args = parse_args()
    input_dir = args.input_img_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(input_dir+'/detected'):
        os.makedirs(input_dir+'/detected')
    eval_imgs = glob.glob(os.path.join(input_dir, '*g'))
    print(eval_imgs)

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    prog_bar = mmcv.ProgressBar(len(eval_imgs))
    for im in eval_imgs:
        detections = mock_detector(model, im, output_dir, input_dir)
        shutil.move(im, im.replace(input_dir,input_dir+'/detected/'))
        prog_bar.update()

if __name__ == '__main__':
    run_detector_on_dataset()
    my_end = time.time()
    print('running time: ', my_end-my_start)
