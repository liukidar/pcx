
import numpy as np
import cv2 as cv
import glob as glob
from IS import *
import argparse
import os,sys

# input image path
in_path = "data/"

# for 10x10
x_cors = [2,36,70,104,138,172,206,240,274,308]
y_cors = [2,36,70,104,138,172,206,240,274,308]
img_size = 32
number_channel = 3

out_path = in_path + "/crops/"
if not os.path.exists(out_path):
    os.makedirs(out_path)

in_list = glob.glob(in_path + "*.png")
count = 0
for img_name in in_list:
    count += 1
    if(number_channel == 1):
        img = cv.imread(img_name,0)
    else:
        img = cv.imread(img_name,1)
    for x in x_cors:
        for y in y_cors:
            img_crop = img[x:x+img_size,y:y+img_size]
            print(img_crop.shape)
            if (number_channel == 1):
                h,w = img_crop.shape
            else:
                h,w,c = img_crop.shape
            if(h != img_size) or (w != img_size):
                print("ERROR!!!")
                exit()

            out_name = out_path +str(count) +"_"+ str(x)+"_"+str(y) + ".png"
            print(out_name)
            cv.imwrite(out_name,img_crop)


parser = argparse.ArgumentParser()
parser.add_argument('--input_npy_file', default=None)
parser.add_argument('--input_image_dir', default=out_path)
parser.add_argument('--input_image_dir_list', default=None)
parser.add_argument('--input_image_superdir', default=None)
parser.add_argument('--image_size', default=1024, type=int)

# Most papers use 50k samples and 10 splits but I don't have that much
# data so I'll use 3 splits for everything
parser.add_argument('--num_splits', default=2, type=int)
parser.add_argument('--tensor_layout', default='NHWC', choices=['NHWC', 'NCHW'])

args = parser.parse_args()
print("read args finished.")
main(args)

