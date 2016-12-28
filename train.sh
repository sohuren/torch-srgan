#/bin/bash
output_hdf5_file="train_data_10.h5"
CUDA_VISIBLE_DEVICES=0 th train.lua -h5_file $output_hdf5_file -num_epoch 50 -loss 'pixel'
#CUDA_VISIBLE_DEVICES=0 th train.lua -h5_file $output_hdf5_file -num_epoch 50 -loss 'percep' -percep_layer 'conv2_2' -use_tanh
#CUDA_VISIBLE_DEVICES=0 th train.lua -h5_file $(output_hdf5_file) -num_epoch 50 -loss 'percep' -percep_layer 'conv5_4' -use_tanh
