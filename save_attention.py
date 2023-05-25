import torch
import os
import numpy as np
from models.vgg_c import vgg19_trans
from glob import glob
from torchvision import transforms
from PIL import Image
import argparse
import cv2
import matplotlib.pyplot as plt

from shutil import copy


args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='model/img_1148.jpg',
                        help='training data directory')
    parser.add_argument('--model-path', default=r'model/best_model.pth',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    torch.backends.cudnn.benchmark = False
    save_dir = os.path.dirname(args.model_path)

    save_dir_viz = os.path.join(save_dir, 'map01')
    if not os.path.exists(save_dir_viz):
        os.makedirs(save_dir_viz)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # im_list = glob(os.path.join(args.data_dir, '*.jpg'))
    im_list = [args.data_dir]

    device = torch.device('cuda')
    model = vgg19_trans(device)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, device))
    epoch_minus = []

    for im_path in im_list:
        # gd_path = im_path.replace('jpg', 'npy')
        # keypoints = np.load(gd_path)
        # name = os.path.basename(im_path).replace('jpg', 'npy')
        # print(name)
        img = Image.open(im_path).convert('RGB')
        w, h = img.size
        # w = w // 2
        # h = h // 2
        # img = img.resize((w,h))
        im = cv2.imread(im_path)

        img = trans(img).unsqueeze_(0)
        inputs = img.to(device)

        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            features = model(inputs)[-1]
            # logf = '_gd_{:.2f}_bpre_{:.2f}'.format(len(keypoints), torch.sum(outputs))
            for layer, feature in enumerate(features):
                mean_fea = torch.min(feature, dim=0)[0]

                # for idx, fea in enumerate(feature.permute(1,0,2)):
                #
                s = 32
                for idx, fea in enumerate(feature):
                    # if idx % 16 == 0 and (idx//(w//32))%8 == 0:
                    # if (idx // (w // 32)) % 8 == 0:
                    if idx % 1 == 0 and layer ==0:
                        fea = fea - mean_fea
                        fea1 = fea.resize(h // s, w // s).detach().cpu().numpy()
                        fea1 = cv2.resize(fea1, (w, h)) / 1.0
                        fea1 = fea1 / np.max(fea1)
                        fea1 = np.clip(fea1, 0.0, 1.0) * 255
                        fea1 = fea1.astype(np.uint8)
                        fea1 = cv2.applyColorMap(fea1, cv2.COLORMAP_JET)

                        final = cv2.addWeighted(im, 0.4, fea1, 0.6, 0)
                        # cv2.circle(final, (idx%(w//32)*32, idx//(w//32)*32), 64, (255,255,255), -1)
                        cv2.circle(final, (idx%(w//s)*s, idx//(w//s)*s), s, (255, 255, 255), -1)

                        fea1log = 'fea_{}_{}2.jpg'.format(idx, layer)
                        cv2.imwrite(os.path.join(save_dir_viz, fea1log), final)

                    # fea1 = fea[1].resize(h//32, w//32).detach().cpu().numpy()
                    # fea1 = cv2.resize(fea1, (w, h)) / 1.0
                    # fea1 = fea1 / np.max(fea1) * 255
                    # fea1 = np.clip(fea1, 0.0, 1.0) * 255
                    # fea1 = fea1.astype(np.uint8)
                    # fea1 = cv2.applyColorMap(fea1, cv2.COLORMAP_JET)
                    # fea1log = 'fea2_{}_{}.jpg'.format(idx, layer)
                    # cv2.imwrite(os.path.join(save_dir_viz, fea1log), fea1)


            # outputs = cv2.resize(outputs, (w, h)) / 1.0
            # # outputs = outputs / np.max(outputs) * 255
            # outputs = np.clip(outputs, 0.0, 1.0) * 255
            # outputs = outputs.astype(np.uint8)
            # outputs = cv2.applyColorMap(outputs, cv2.COLORMAP_JET)
            #
            #
            # copy(im_path,
            #      im_path.replace(os.path.dirname(im_path), save_dir_viz))
            # cv2.imwrite(os.path.join(save_dir_viz, logf+'_d.jpg'), outputs)

