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
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
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

    parser.add_argument('--images_path', default='',
                        help='path where the images loaded')                   

    parser.add_argument('--predicts_txt_dir', default='',
                        help='path where the predict saved')

    parser.add_argument('--predicts_point_dir', default='',
                        help='path where the predict points saved')

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
    all_point=[]
    c=1
    img_path = args.images_path
    labels_path=os.path.join(args.images_path,'Labels.txt')
    labels=open(labels_path,'r')
    groundtruth=[]
    for x in labels:
      groundtruth.append(int(x))
    # load the images
    
    #for filename in os.listdir(img_path)[:3]:
    for filename in range(1,201):
        filename=f"{filename:03}"
        fname=np.copy(filename)
        fname=str(fname)+'.jpg'
        filename=int(filename)
        #img = cv2.imread(os.path.join(folder,filename))
        img_raw = Image.open(os.path.join(img_path,fname)).convert('RGB')
        # round the size
        width, height = img_raw.size
        if width>2000 or height>2000:
          r=width/height
          width=2000
          height=int(2000/r)
        new_width = width // 128 * 128
        new_height = height // 128 * 128

        print('img{}'.format(filename))
        img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
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
        
        outputs_points = outputs['pred_points'][0]
        # draw the predictions
        if new_width<1000:
          size = 1
        elif new_width>1000 and new_width<1500:
          size=2
        elif new_width>1500:
          size=3
        else:
          size=1
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)

        all_point.append(points)
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        # save the visualized image
        predict=np.shape(points)[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 0.5
        # Yellow color in BGR
        color = (0,250,255)
        # Line thickness of 2 px
        thickness = 2
        img_to_draw = cv2.putText(img_to_draw,'predict={}'.format(predict),(50,50),font,fontScale, color, thickness, cv2.LINE_AA)
        gr=groundtruth[filename-1]
        img_to_draw = cv2.putText(img_to_draw,'groundtruth={}'.format(gr),(50,70),font,fontScale, color, thickness, cv2.LINE_AA)
        
        if not os.path.isdir(args.output_dir):
          os.makedirs(args.output_dir)

        cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(filename)), img_to_draw)
        if c==1:
          file = open(args.predicts_txt_dir, "w+")
        c+=1
        file.write("img{}  pre={}  gr={}\n".format(int(filename),predict,gr))
        
        if c==201:
          all_point=np.array(all_point)
          np.save(os.path.join(args.predicts_point_dir,'points.npy'),all_point)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)