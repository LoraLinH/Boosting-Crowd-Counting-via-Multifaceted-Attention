from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('images', 'gt').replace('.jpg', '.txt')
    points = []
    dis = []
    with open (mat_path, 'r') as f:
        while True:
            point = f.readline()
            if not point:
                break
            point = point.split(' ')[:-1]
            points.append([float(point[0]), float(point[1])])
            dis.append([float(point[3])])
    points = np.array(points)
    dis = np.array(dis)
    if len(points>0):
        idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
        points = points[idx_mask]
        dis = dis[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
        dis = dis * rr
    return Image.fromarray(im), points, dis


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='E:\Dataset\Counting\jhu_crowd_v2.0',
                        help='original data directory')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048

    for phase in ['train', 'val', 'test']:
        sub_dir = os.path.join(args.origin_dir, phase)
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        im_list = glob(os.path.join(os.path.join(sub_dir, 'images'), '*jpg'))
        for im_path in im_list:
            name = os.path.basename(im_path)
            print(name)
            im, points, dis = generate_data(im_path)
            if phase == 'train':
                if len(points)>0:
                    points = np.concatenate((points, dis), axis=1)
            im_save_path = os.path.join(sub_save_dir, name)
            im.save(im_save_path, quality=95)
            gd_save_path = im_save_path.replace('jpg', 'npy')
            np.save(gd_save_path, points)
