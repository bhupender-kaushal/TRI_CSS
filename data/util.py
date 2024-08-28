import os
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as v2

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)
    def _augment(img):
        if hflip:
            img = img[:, ::-1]
        if vflip:
            img = img[::-1, :]
        if rot90:
            img = img.transpose(1, 0)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


def transform2tensor(img, min_max=(0, 1),eq=False):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float() #.unsqueeze(0)
    # to range min_max
    if eq:
        img=(F.equalize((img*255).type(torch.uint8)).float())/255.
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img

def _random_crop(data):
    inputSize=256
    img_list=[0,0]

    nh, nw = data[0].shape
    opt1, opt2 = 0, 0
    if nh > inputSize:
        opt1 = np.random.randint(0, nh-inputSize)
    if nw > inputSize:
        opt2 = np.random.randint(0, nw-inputSize)

    data1 = data[0][opt1:opt1+inputSize, opt2:opt2+inputSize]
    data2 = data[1][opt1:opt1+inputSize, opt2:opt2+inputSize]
    img_list[0]=data1
    img_list[1]=data2
    # return ret_img
    return img_list

def transform_augment(img_list, split='val', min_max=(0, 1), eq=False):
    ret_img = []
    # img_list=_random_crop(img_list)
    img_list = augment(img_list, split=split)
    for img in img_list:
        img = transform2numpy(img)
        img = transform2tensor(img, min_max, eq=False)
        # img=F.resize(img,(512,512))
        # # # # cropper = v2.RandomCrop(size=(256, 256))
        # # # # img=cropper(img)
        # img=F.center_crop(img,512)
        ret_img.append(img)
    return ret_img