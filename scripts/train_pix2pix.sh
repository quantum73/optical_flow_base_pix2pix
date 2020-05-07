set -ex
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0

python train.py --dataroot ./datasets/optical_flow --name optical_flow --model pix2pix --netG unet_256 --direction AtoB --dataset_mode aligned --input_nc 8 --output_nc 2 --n_layers_D 6