set -ex
python train.py --dataroot ./datasets/optical_flow --name optical_flow --model pix2pix --dataset_mode aligned --netG unet_256 --direction AtoB --n_layers_D 6 --gpu -1
