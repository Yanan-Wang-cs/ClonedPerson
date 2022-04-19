from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math
import numpy as np


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, crop_factor=0.8, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.crop_factor = crop_factor
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.crop_factor ** 2, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)


class RandomErasing(object):
    def __init__(self, EPSILON=0.5, mean=[0.485, 0.456, 0.406]):
        self.EPSILON = EPSILON
        self.mean = mean

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(0.02, 0.2) * area
            aspect_ratio = random.uniform(0.3, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]

                return img

        return img


class RandomOcclusion(object):
    def __init__(self, min_size=0.2, max_size=1):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        if self.max_size == 0:
            return img
        H = img.height
        W = img.width
        S = min(H, W)
        s = np.random.randint(max(1, int(S * self.min_size)), int(S * self.max_size))
        x0 = np.random.randint(W - s)
        y0 = np.random.randint(H - s)
        x1 = x0 + s
        y1 = y0 + s
        block = Image.new('RGB', (s, s), (255, 255, 255))
        img.paste(block, (x0, y0, x1, y1))
        return img


class MyRandomCrop_side(object):
    def __init__(self, side_rate=0.0):
        self.side_rate=side_rate

    def __call__(self,img):
        W = img.width
        H = img.height
        random_index = random.randint(0,2)
        x1 = random.randint(0, int(self.side_rate*W))
        x2 = W - random.randint(0, int(self.side_rate*W))
        if random_index == 0:
            x2 = W
        elif random_index == 1:
            x1 = 0
        img = img.crop((x1,0,x2,H))
        #img.save('/home/yw1/crop/'+str(random_index)+'_'+str(random.randint(0,100000000))+'.jpg')
        return img


class MyRandomCrop_bottom(object):
    def __init__(self, crop_rate=0.0):
        self.crop_rate=crop_rate

    def __call__(self,img):
        if self.crop_rate == 0:
            return img
        randomnum = random.randint(0,100)/100
        if randomnum > self.crop_rate:
            return img
        H = img.height
        W = img.width
        y2 = H - random.randint(0,int(0.5*H))
        img = img.crop((0,0,W,y2))
        return img
