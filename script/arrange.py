# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DVD-Dataset')
    parser.add_argument('--data_path', type=str, default='../data/DeepVideoDeblurring_Dataset',
                        help='path of dataset')
    parser.add_argument('--out_path', type=str, default='../data/DVD',
                        help='path to save output dataset')
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    # Define output path
    test_gt = osp.join(args.out_path, 'Test/gt')
    test_blur = osp.join(args.out_path, 'Test/blur')
    train_gt = osp.join(args.out_path, 'Train/gt')
    train_blur = osp.join(args.out_path, 'Train/blur')

    # Select videos for testing
    test_list = ['720p_240fps_2', 'IMG_0003', 'IMG_0021', 'IMG_0030', 'IMG_0031',
                 'IMG_0032', 'IMG_0033', 'IMG_0037', 'IMG_0039', 'IMG_0049']

    if not os.path.exists(test_gt):
        os.makedirs(test_gt)
    if not os.path.exists(test_blur):
        os.makedirs(test_blur)
    if not os.path.exists(train_gt):
        os.makedirs(train_gt)
    if not os.path.exists(train_blur):
        os.makedirs(train_blur)

    data_path = os.path.join(args.data_path, 'quantitative_datasets')
    folder_list = [x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x))]

    for folder in folder_list:
        blur_path_src = os.path.join(data_path, folder, 'input')
        gt_path_src = os.path.join(data_path, folder, 'GT')
        # move to Test
        if folder in test_list:
            blur_path_tar = os.path.join(test_blur, folder)
            gt_path_tar = os.path.join(test_gt, folder)

        # move to Train
        else:
            blur_path_tar = os.path.join(train_blur, folder)
            gt_path_tar = os.path.join(train_gt, folder)

        if os.path.exists(blur_path_src):
            cmd = 'cp -r {} {}'.format(blur_path_src, blur_path_tar)
            os.system(cmd)

        if os.path.exists(gt_path_src):
            cmd = 'cp -r {} {}'.format(gt_path_src, gt_path_tar)
            os.system(cmd)
