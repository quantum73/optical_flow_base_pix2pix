"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    left_dir = os.path.join(dir, 'LeftImages')
    right_dir = os.path.join(dir, 'RightImages')
    flow_noise_dir = os.path.join(dir, 'OpticalFlowNoise')
    flow_gt_dir = os.path.join(dir, 'OpticalFlowGT')

    for root, _, fnames in sorted(os.walk(left_dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path_left = os.path.join(root, fname)
                path_right = os.path.join(right_dir, fname)
                path_flow_noise = os.path.join(flow_noise_dir, fname.replace('.png', '.npz'))
                images.append({'A':path_left, 'B':'', 'C':path_right, 'D':path_flow_noise})
    
    j = 0
    for root, _, fnames in sorted(os.walk(flow_gt_dir)):
        for fname in sorted(fnames):
            # if is_image_file(fname):                
            path_real = os.path.join(root, fname)
            images[j]['B'] = path_real
            j += 1                
    
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
