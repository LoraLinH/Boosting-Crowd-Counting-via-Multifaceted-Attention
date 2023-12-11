import torch
import os
import numpy as np
from models.vgg_c import vgg19_trans
from glob import glob
from torchvision import transforms
from PIL import Image
import argparse
import cv2
from shutil import copy


args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default=r'E:\Dataset\Counting\UCF-Train-Val-Test\test',
                        help='training data directory')
    parser.add_argument('--model-path', default='model/best_model.pth',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    torch.backends.cudnn.benchmark = False
    save_dir = os.path.dirname(args.model_path)


    save_dir_d = os.path.join(save_dir, 'density')
    if not os.path.exists(save_dir_d):
        os.makedirs(save_dir_d)

    save_dir_viz = os.path.join(save_dir, 'vis')
    if not os.path.exists(save_dir_viz):
        os.makedirs(save_dir_viz)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    im_list = glob(os.path.join(args.data_dir, '*.jpg'))


    device = torch.device('cuda')
    model = vgg19_trans()
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, device))
    epoch_minus = []
    num = 0

    for im_path in im_list:
        gd_path = im_path.replace('jpg', 'npy')
        keypoints = np.load(gd_path)
        name = os.path.basename(im_path).split('.')
        # print(name)
        img = Image.open(im_path).convert('RGB')
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        img = Image.fromarray(img_np)
        img = trans(img).unsqueeze_(0)
        inputs = img.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)[0]
            logf = '{}gd_{:.2f}_bpre_{:.2f}.jpg'.format(num, len(keypoints), torch.sum(outputs))
            num += 1
            # logf = 'bpre_{:.2f}'.format(torch.sum(outputs))
            outputs = outputs.detach().cpu().numpy()[0][0]
            # np.save(os.path.join(save_dir_d, name), outputs)


            outputs = cv2.resize(outputs, (w, h)) / 1.0
            # outputs = outputs / np.max(outputs) * 255
            outputs = np.clip(outputs, 0.0, 1.0) * 255
            outputs = outputs.astype(np.uint8)
            outputs = cv2.applyColorMap(outputs, cv2.COLORMAP_JET)


            # copy(im_path,
            #      im_path.replace(os.path.dirname(im_path), save_dir_viz))
            # cv2.imwrite(os.path.join(save_dir_viz,
            #                          im_path.replace('.JPG', logf+'_d.jpg')), outputs)
            cv2.imwrite(os.path.join(save_dir_viz, logf), outputs)

