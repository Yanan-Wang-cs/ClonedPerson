from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob


class Cluster(object):

    def __init__(self, root,img_path, combine_all=True):

        self.images_dir = osp.join(root)
        self.img_path = img_path
        self.train_path = self.img_path
        self.gallery_path = ''
        self.query_path = ''
        self.train = []
        self.gallery = []
        self.query = []
        self.num_train_ids = 0
        self.has_time_info = False
        self.load()

    def preprocess(self):
        fpaths = sorted(glob(osp.join(self.images_dir, self.train_path,'*')))
        data = []
        all_pids = {}
        for fpath in fpaths:
            fname=osp.basename(fpath)
            pid = len(all_pids)
            all_pids[len(all_pids)]=len(all_pids)
            camid = 0
            time = 0
            data.append((fname, pid, camid, time))
        return data, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess()

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  all    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
