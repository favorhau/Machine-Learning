import random
import numpy as np
import argparse
from PIL import Image
from src import util as iu
from src import dp_pixel as dp
from src import dataset as db
from src.resize import Resize
from src.pixelate import Pixelate

resize_f = Resize.pad_image
pixelate_f = Pixelate.pytorch
def save_npy(images,out_path):
    np.save(out_path,images)

def dp_pixelate_images(images, target_h, target_w, m, eps, out_path):
    noisy_images = [dp.dp_pixelate(I,target_h, target_w, m, eps, resize_f=resize_f, 
                                   pixelate_f=pixelate_f) for I in images]
    save_npy(noisy_images[0],out_path)

def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual losses', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw_path', default='raw.npy',help='the orgin path of the features .npy')
    parser.add_argument('--out_path', default='dp_features.npy',help='the output path of the features .npy')
    parser.add_argument('--target_h', default=18,help='the height of the output images')
    parser.add_argument('--target_w', default=512,help='the weight of the output images')
    parser.add_argument('--m', default=1, help='the argument m that refers to the privacy standard')
    parser.add_argument('--eps', default=10000, help='the argument eps that refers to the privacy standard')

    args = parser.parse_args()
    images =np.array(np.load(args.raw_path))
    dp_pixelate_images([images],args.target_h,args.target_w,args.m,args.eps,args.out_path)
    print('the laplace-added features representation is at {}'.format(args.out_path))


if __name__ == '__main__':
    main()
