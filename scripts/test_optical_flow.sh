set -ex
python test.py --dataroot ./datasets/optical_flow --name optical_flow --model pix2pix --dataset_mode aligned --netG unet_256 --direction AtoB --input_nc 8 --output_nc 2 --n_layers_D 6