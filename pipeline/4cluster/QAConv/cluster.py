from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import os
import time
import shutil

import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch import nn
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

import sys
sys.path.append('../../../Pub/QAConv/')

from reid import datasets
from reid.models import resmap
from reid.models.qaconv import QAConv
from reid.utils.data import transforms as T
from reid.utils.serialization import load_checkpoint
from reid.loss.pairwise_matching_loss import PairwiseMatchingLoss
from reid.evaluators import extract_features, pairwise_distance

from reid.utils.data.preprocessor import Preprocessor
from reid.evaluators import extract_features, pairwise_distance, reranking


def calc_distance( dataset, img_path, transformer, model, matcher, rerank=True, gal_batch_size=4, prob_batch_size=4096):
    data_loader = DataLoader(
        Preprocessor(dataset, img_path, transform=transformer),
        batch_size=64, num_workers=8,
        shuffle=False, pin_memory=True)

    features, _ = extract_features(model, data_loader)
    features = torch.cat([features[fname].unsqueeze(0) for fname, _, _, _ in dataset], 0)

    print('Compute distance...', end='\t')
    start = time.time()
    dist = pairwise_distance(matcher, features, features, gal_batch_size, prob_batch_size)
    print('Time: %.3f seconds.' % (time.time() - start))

    if rerank:
        print('Rerank...', end='\t')
        start = time.time()
        with torch.no_grad():
            dist = torch.cat((dist, dist))
            dist = torch.cat((dist, dist), dim=1)
            dist = reranking(dist, len(dataset))
            dist = torch.from_numpy(dist).cuda()
        print('Time: %.3f seconds.' % (time.time() - start))

    return dist


def get_data(dataname, data_dir,imgfile, args):
    print(dataname, data_dir,imgfile, args)
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root,imgfile, combine_all=args.combine_all)

    train_path = osp.join(dataset.images_dir, dataset.train_path)

    test_transformer = T.Compose([
        T.Resize((args.height, args.width), interpolation=3),
        T.ToTensor(),
    ])

    return dataset.train, train_path, test_transformer

def get_dist_average(dist_numpy):
    row, col = np.diag_indices_from(dist_numpy)
    dist_numpy[row, col] = 0
    dist_average = np.average(dist_numpy, axis=1)
    return dist_average

def main(args):
    for file in os.listdir(osp.join(args.data_dir, args.dataset)):
        cudnn.deterministic = False
        cudnn.benchmark = True

        # Create model
        model = resmap.create(args.arch, ibn_type=args.ibn, final_layer=args.final_layer, neck=args.neck).cuda()
        num_features = model.num_features
        feamap_factor = {'layer2': 8, 'layer3': 16, 'layer4': 32}
        hei = args.height // feamap_factor[args.final_layer]
        wid = args.width // feamap_factor[args.final_layer]
        matcher = QAConv(num_features, hei, wid).cuda()

        for arg in sys.argv:
            print('%s ' % arg, end='')
        print('\n')

        dataset, img_path, transformer = get_data(args.dataset, args.data_dir, file, args)

        # Criterion
        criterion = PairwiseMatchingLoss(matcher).cuda()

        # Load from checkpoint
        print('Loading checkpoint...')
        print('###', osp.join(args.save_path, 'checkpoint.pth.tar'))
        checkpoint = load_checkpoint(osp.join(args.save_path, 'checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['model'])
        criterion.load_state_dict(checkpoint['criterion'])

        model = nn.DataParallel(model).cuda()

        dist = calc_distance(dataset, img_path, transformer, model, matcher, True, args.gal_batch_size, args.prob_batch_size)
        dist_numpy = dist.cpu().numpy()
        folder = img_path.split('/').pop()
        savepath = args.save_data_path + folder + '/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        if dist.size()[0] >= 2:
            # DBSCAN cluster
            tri_mat = np.triu(dist.cpu(), 1)  # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
            tri_mat = np.sort(tri_mat, axis=None)

            rho = args.rho
            top_num = np.round(rho * tri_mat.size).astype(int)
            # eps = tri_mat[:top_num].mean()
            eps = args.eps
            savepath_new = savepath + str(eps) + '/'
            print('savepath:', savepath_new)
            if not os.path.exists(savepath_new):
                os.makedirs(savepath_new)
            print('EPS for cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=args.min_samples, metric='precomputed', n_jobs=-1)

            print('Clustering and labeling...')
            labels = cluster.fit_predict(dist.cpu())
            for i in range(-1, np.max(labels)+1):
                index_list = np.where(labels == i)[0]
                if len(index_list) > 0:
                    reg1=dist_numpy[np.ix_(index_list, index_list)]
                    pic1 = np.array(dataset)[index_list, 0]
                    dist_average=get_dist_average(reg1)
                    current_savepath = savepath_new + folder+'_'+str(i) + '/'
                    if not os.path.exists(current_savepath):
                        os.makedirs(current_savepath)
                    for j in range(0, len(pic1)):
                        print(current_savepath +str(dist_average[j])+ '_' +pic1[j])
                        shutil.copyfile(img_path+'/'+pic1[j],  current_savepath +str(dist_average[j])+ '_' +pic1[j])
            num_ids = len(set(labels)) - (1 if -1 in labels else 0)
            print('DBSCAN: clustered into {} classes.'.format(num_ids))

            labels = np.array(labels)

            noisy_samples = (labels == -1).sum()
            core_samples = len(dataset) - noisy_samples
            print('Core samples: %d. Noisy samples: %d. Average samples per cluster: %d.\n' % (core_samples, noisy_samples, core_samples // num_ids))

        else:
            average_dist = get_dist_average(dist_numpy)
            for i in range(0, len(dataset)):
                shutil.copyfile(img_path+'/'+dataset[i][0],  savepath +str(average_dist[i])+ '_' +dataset[i][0])

            print(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QAConv")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='unity', choices=datasets.names(),help="the training dataset")
    parser.add_argument('--combine_all', action='store_true', default=False,help="combine all data for training, default: False")
    parser.add_argument('-j', '--workers', type=int, default=8,help="the number of workers for the dataloader, default: 8")
    parser.add_argument('--height', type=int, default=384, help="height of the input image, default: 384")
    parser.add_argument('--width', type=int, default=128, help="width of the input image, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=resmap.names(),help="the backbone network, default: resnet50")
    parser.add_argument('--final_layer', type=str, default='layer3', choices=['layer2', 'layer3', 'layer4'],help="the final layer, default: layer3")
    parser.add_argument('--neck', type=int, default=64,help="number of channels for the final neck layer, default: 64")
    parser.add_argument('--ibn', type=str, choices={'a', 'b'}, default='b', help="IBN type. Choose from 'a' or 'b'. Default: None")
    parser.add_argument('--gal_batch_size', type=int, default=4, help="x")

    parser.add_argument('--prob_batch_size', type=int, default=4096, help="x")
    parser.add_argument('--rho', type=float, default=0.0016)
    parser.add_argument('--min_samples', type=int, default=5)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'),help="the path to the image data")
    parser.add_argument('--eps', type=float, default=0.4)
    parser.add_argument('--save_path', type=str, metavar='PATH', default=osp.join(working_dir, ''),help="x")
    parser.add_argument('--save_data_path', type=str, metavar='PATH', default=osp.join(working_dir, 'clonedperson_deepFashion'), help="x")


    main(parser.parse_args())
