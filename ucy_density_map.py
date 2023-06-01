import torch
import os
import numpy as np
from models.vgg_c import vgg19_trans
import argparse
import math
import cv2 as cv
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--stream', default='datasets/video.avi',help='training data directory')
    parser.add_argument('--model', default='models/UCF.pth',help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    device = torch.device('cpu')
    model = vgg19_trans()
    model.to(device)
    model.eval()

    model.load_state_dict(torch.load(args.model, device))
    epoch_minus = []

    stream = cv.VideoCapture(args.stream)
    imWidth = int(stream.get(cv.CAP_PROP_FRAME_WIDTH))
    imHeight = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    print("ImgW:{}, ImgH:{}".format(imWidth,imHeight))
    streamWritter = cv.VideoWriter("output.avi", cv.VideoWriter_fourcc(*'MJPG'), 10, (imWidth, imHeight))

    if (stream.isOpened()== False):
        print("Error opening stream")

    frameTensor = torch.zeros(1, 3, imHeight, imWidth)

    trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    i = 0
    while(stream.isOpened()):
        ret, frame = stream.read()
        pilFrame = Image.fromarray(frame)
        tmp = trans(pilFrame)
        frameTensor[0,:,:,:] = tmp
        frameTensor.to(device)
        with torch.set_grad_enabled(False):
            output = model(frameTensor)[0]
            denMap = output[0][0].detach().cpu().numpy()

        #streamWritter.write(denMap*255)
        if i < 10:
            #cv.imwrite("densityMap"+str(i)+".png", denMap*255)
            plt.imsave("densityMap"+str(i)+".png", denMap*255)
        else:
            break
        i +=1

    stream.release()
    cv.destroyAllWindows()