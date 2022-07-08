import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm as CM
from PIL import Image
import scipy
import scipy.io as io
from scipy import spatial
from scipy.ndimage.filters import gaussian_filter
import h5py

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for density_map', add_help=False)

    parser.add_argument('--images_path', default='')
    
    parser.add_argument('--points_path', default='')

    parser.add_argument('--method', default='fixed',
                        help='fixed , knn')
    
    parser.add_argument('--output_dir', default='',
                        help='path where to save')

    return parser


def generate_density_map_with_fixed_kernel(images_path,all_points,density_map_dir):
    '''
    img: input image.
    points: annotated pedestrian's position like [row,col]
    kernel_size: the fixed size of gaussian kernel, must be odd number.
    sigma: the sigma of gaussian kernel.
    return:
    d_map: density-map we want
    '''
    
    def guassian_kernel(size,sigma):
        rows=size[0] # mind that size must be odd number.
        cols=size[1]
        mean_x=int((rows-1)/2)
        mean_y=int((cols-1)/2)

        f=np.zeros(size)
        for x in range(0,rows):
            for y in range(0,cols):
                mean_x2=(x-mean_x)*(x-mean_x)
                mean_y2=(y-mean_y)*(y-mean_y)
                f[x,y]=(1.0/(2.0*np.pi*sigma*sigma))*np.exp((mean_x2+mean_y2)/(-2.0*sigma*sigma))
        return f
    c=1
    #for filename in os.listdir(images_path):
    for filename in range(1,201):

        pts=all_points[filename-1]
        points=[]
        for p in pts:
          points.append([p[1],p[0]]) #convert (col,row) to (row,col)

        filename=f"{filename:03}"
        fname=np.copy(filename)
        fname=str(fname)+'.jpg'
        filename=int(filename)
        #img = Image.open(os.path.join(images_path,filename)).convert('RGB')
        img=plt.imread(os.path.join(images_path,fname))

        [rows,cols]=[img.shape[0],img.shape[1]]
        if rows<600:
          kernel_size=200
          sigma=30
        elif rows>600 and rows<1200:
          kernel_size=300
          sigma=40

        elif rows>1200:
          kernel_size=400
          sigma=60
        else:
          kernel_size=250
          sigma=35

        d_map=np.zeros([rows,cols])
        f=guassian_kernel([kernel_size,kernel_size],sigma) # generate gaussian kernel with fixed size.
        normed_f=(1.0/f.sum())*f # normalization for each head.
        
        if len(points)==0:
            return d_map
        else:
            for p in points:
                r,c=int(p[0]),int(p[1])
                if r>=rows or c>=cols:
                    continue
                for x in range(0,f.shape[0]):
                    for y in range(0,f.shape[1]):
                        if x+((r+1)-int((f.shape[0]-1)/2))<0 or x+((r+1)-int((f.shape[0]-1)/2))>rows-1 \
                        or y+((c+1)-int((f.shape[1]-1)/2))<0 or y+((c+1)-int((f.shape[1]-1)/2))>cols-1:
                            continue
                        else:
                            d_map[x+((r+1)-int((f.shape[0]-1)/2)),y+((c+1)-int((f.shape[1]-1)/2))]+=normed_f[x,y]
                          
        print(fname)
        #cv2.imwrite(os.path.join(density_map_dir,'density_map{}.png'.format(filename)),d_map)
        plt.imsave(os.path.join(density_map_dir,'density_map{}.png'.format(filename)),d_map, cmap=CM.jet)


def gaussian_filter_density(images_path,all_points,density_map_dir):

    for filename in range(1,201):

        points=all_points[filename-1]

        filename=f"{filename:03}"
        fname=np.copy(filename)
        fname=str(fname)+'.jpg'
        filename=int(filename)
        #img = Image.open(os.path.join(images_path,filename)).convert('RGB')
        img=plt.imread(os.path.join(images_path,fname))

        img_shape=[img.shape[0],img.shape[1]]
        print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
        density = np.zeros(img_shape, dtype=np.float32)
        gt_count = len(points)
        if gt_count == 0:
            return density

        leafsize = 2048
        # build kdtree
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(points, k=4)

        print ('generate density...')
        for i, pt in enumerate(points):
            pt2d = np.zeros(img_shape, dtype=np.float32)
            if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
                pt2d[int(pt[1]),int(pt[0])] = 1.
            else:
                continue
            if gt_count > 1:
                sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            else:
                sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        print ('done.')
        plt.imsave(os.path.join(density_map_dir,'density_map{}.png'.format(filename)),density, cmap=CM.jet)
    

# test code
if __name__=="__main__":
    # show an example to use function generate_density_map_with_fixed_kernel.
    parser = argparse.ArgumentParser('density map', parents=[get_args_parser()])
    args = parser.parse_args()

    imgs_path=args.images_path
    All_points=np.load(args.points_path,allow_pickle=True)
    method=args.method
    density_dir=args.output_dir
    #kernel_size=args.Kernel_size
    #sigma=args.Sigma

    if not os.path.isdir(density_dir):
          os.makedirs(density_dir)

    if method=='fixed':
      generate_density_map_with_fixed_kernel(imgs_path,All_points,density_dir)
    elif method=='knn':
      gaussian_filter_density(imgs_path,All_points,density_dir)