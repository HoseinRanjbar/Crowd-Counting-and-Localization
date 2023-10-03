import argparse
import datetime
import random
import time
from pathlib import Path
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import cv2
from dataset import build_dataset
from engine import *
from models import build_model
import os
import warnings
from google.colab.patches import cv2_imshow

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--threshold', default=0.5,type=float)

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")


    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')

    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--video_path', default='',
                        help='path where the images loaded')                   

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser


def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    video_path = args.video_path
    vidcap = cv2.VideoCapture(video_path)
    sec = 0
    frameRate = 0.05 #//it will capture image in each 0.1 second
    count=1
    flag=True
    c=1
    while flag:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames==True:
            img_raw=np.copy(image)
        else:
          out.release()
          cv2.destroyAllWindows()
          vidcap.release()
          break
        img_raw = Image.fromarray(img_raw)
        width, height = img_raw.size
        if width>2000 or height>2000:
              r=width/height
              width=2000
              height=int(2000/r)
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        
        img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        if c==1:
              size=img_raw.size
              out = cv2.VideoWriter(args.output_dir,cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
              c+=1
        # pre-proccessing
        img = transform(img_raw)
            

        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)
            # run inference
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        threshold = args.threshold
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        size=2
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
            
        out.write(img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
