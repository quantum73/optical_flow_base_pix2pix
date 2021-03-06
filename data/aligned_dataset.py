import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from torchvision import transforms


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = make_dataset(self.dir_AB, opt.max_dataset_size) # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
            
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        """
        
        AB_paths = self.AB_paths[index]
        left_img = Image.open(AB_paths['A']).convert('RGB')
        right_img = Image.open(AB_paths['C']).convert('RGB')
        flow_noise = np.load(AB_paths['D'])['arr_0']
        flow_gt = np.load(AB_paths['B'])['arr_0']
        
        crop_size = 512
        img_w, img_h = left_img.size
        min_dim = min(img_w, img_h)
        crop_size = min(crop_size, min_dim)
        self.crop_indices = transforms.RandomCrop.get_params(left_img, output_size=(crop_size, crop_size))
        i, j, h, w = self.crop_indices
        
        left_img = np.array(left_img)
        right_img = np.array(right_img)
        left_img = left_img[i:i+h, j:j+w, :]
        right_img = right_img[i:i+h, j:j+w, :]
        flow_noise = flow_noise[i:i+h, j:j+w, :]

        # normalize data to [-1, 1]
        left_img = left_img / 255.0 * 2.0 - 1.0
        right_img = right_img / 255.0 * 2.0 - 1.0
        flow_gt = flow_gt / 50
        flow_gt = np.clip(flow_gt, -1.0, 1.0)
        flow_noise = flow_noise / 50
        flow_noise = np.clip(flow_noise, -1.0, 1.0)

        A = np.dstack((left_img, right_img, flow_noise))
        B = flow_gt[i:i+h, j:j+w, :]
        A = np.transpose(A, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        
        return {'A': A, 'B': B, 'A_paths': AB_paths}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
