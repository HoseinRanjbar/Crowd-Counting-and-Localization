import argparse
import numpy as np
import cv2
import os
from scipy.io import loadmat

def get_args_parser():
    parser = argparse.ArgumentParser('dataset preparation', add_help=False)
    
    parser.add_argument('--application', default='both',
                        help='make_list , convert_gr_format or both')

    parser.add_argument('--groundtruth_dir', default='',
                        help='path where to load groundtruth')
    
    parser.add_argument('--groundtruth_txt_path', default='',
                        help='path where to save groundtruth information with .txt format')
    
    parser.add_argument('--list_dir', default='',
                        help='path where to save')
    
    parser.add_argument('--images_path', default='')
    
    return parser

def main(args):
  gt_path=args.groundtruth_dir
  gt_txt_path=args.groundtruth_txt_path
  list_path=args.list_dir
  imgs_path=args.images_path
  app=args.application

  if app=='convert_gr_format':
    #convert grandtruth information from .mat to .txt
    for filename in os.listdir(gt_path):
      gt = loadmat(os.path.join(gt_path,filename))
      info=(gt['image_info'])[0][0][0][0][0]
      file = open(os.path.join(gt_txt_path,'{}.txt'.format(filename[:-4])), "w+")
      for i in range(np.shape(info)[0]):
        x,y=info[i][0],info[i][1]
        file.write("{} {}\n".format(x,y))

  elif app=='make_list':
    #Creat list of images and grandtruth for training
    file = open(os.path.join(list_path,'train_list.txt'), "w+")
    for filename in os.listdir(imgs_path):
      file.write("images/{} gt_txt/GT_{}.txt\n".format(filename,filename[:-4]))
  
  else:
    #convert grandtruth information from .mat to .txt
    for filename in os.listdir(gt_path):
      gt = loadmat(os.path.join(gt_path,filename))
      info=(gt['image_info'])[0][0][0][0][0]
      file = open(os.path.join(gt_txt_path,'{}.txt'.format(filename[:-4])), "w+")
      for i in range(np.shape(info)[0]):
        x,y=info[i][0],info[i][1]
        file.write("{} {}\n".format(x,y))

    #Creat list of images and grandtruth for training
    file = open(os.path.join(list_path,'train_list.txt'), "w+")
    for filename in os.listdir(imgs_path):
      file.write("images/{} gt_txt/GT_{}.txt\n".format(filename,filename[:-4]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('dataset preparation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

