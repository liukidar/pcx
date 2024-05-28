import os,sys
import imageio
import numpy as np
import argparse
import math
from model import ResNet18
import torchvision.transforms as transforms
import torch
import cv2 as cv
import glob as glob
from numpy import clip

GPUID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
print ("PACKAGES LOADED")

def load_images(args, image_dir):
    images = []
    for fn in os.listdir(image_dir):
        ext = os.path.splitext(fn)[1].lower()
        img_path = os.path.join(image_dir, fn)
        img = imageio.imread(img_path)
        # calculate per-channel means and standard deviations
        means = img.mean(axis=(0, 1), dtype='float64')
        stds = img.std(axis=(0, 1), dtype='float64')
        # per-channel standardization of pixels
        pixels = (img - means) / stds
        pixels = clip(pixels, -1.0, 1.0)
        images.append(pixels)
    return images


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.expand_dims(e_x.sum(axis=1), axis=1)   # only difference


def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_inception_score(args, images):
    splits = args.num_splits
    inps = []
    input_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(args.batch_size)))
    n_preds = 0

    net = ResNet18().cuda()
    net.load_state_dict(torch.load(args.model_dir))
    print("load model successfully")

    for i in range(n_batches):
        sys.stdout.write(".")
        sys.stdout.flush()
        inp = inps[(i * args.batch_size):min((i + 1) * args.batch_size, len(inps))]
        inp = np.concatenate(inp, 0)
        inp = torch.from_numpy(np.rollaxis(inp,3,1)).cuda()
        outputs = net(inp)
        pred = outputs.data.tolist()
        #pred = softmax(pred)
        preds.append(pred)
        n_preds += outputs.shape[0]
    preds = np.concatenate(preds, 0)
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    mean_, std_ = preds2score(preds, splits)
    return mean_, std_

def main(args):
  images = load_images(args, args.input_image_dir)
  mean, std = get_inception_score(args, images)
  print('\nInception mean: ', mean)
  print('Inception std: ', std)


def crop10x10(in_path, out_path):
    # svhn

    # 10x10
    x_cors = [2, 36, 70, 104, 138, 172, 206, 240, 274, 308]
    y_cors = [2, 36, 70, 104, 138, 172, 206, 240, 274, 308]
    img_size = 32
    number_channel = 3

    print(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    in_list = glob.glob(in_path + "*.png")
    count = 0
    for img_name in in_list:
        count += 1
        if (number_channel == 1):
            img = cv.imread(img_name, 0)
        else:
            img = cv.imread(img_name, 1)
        for x in x_cors:
            for y in y_cors:
                img_crop = img[x:x + img_size, y:y + img_size]
                #print(img_crop.shape)
                if (number_channel == 1):
                    h, w = img_crop.shape
                else:
                    h, w, c = img_crop.shape
                if (h != img_size) or (w != img_size):
                    print("ERROR!!!")
                    exit()

                out_name = out_path + str(count) + "_" + str(x) + "_" + str(y) + ".png"
                #print(out_name)
                cv.imwrite(out_name, img_crop)

if __name__ == '__main__':
    in_path = "./data/"
    out_path = in_path + "/crops/"
    crop10x10(in_path, out_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_dir', default=out_path)
    parser.add_argument('--model_dir', default="checkpoints/svhn_model_10.ckpt")
    parser.add_argument('--img_size', default=32)
    parser.add_argument('--batch_size', default=100)
    parser.add_argument('--channel', default=3)
    parser.add_argument('--num_splits', default=10)
    args = parser.parse_args()
    main(args)

