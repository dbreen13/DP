#!/bin/bash
#train limits

#resnet18 cifar10
#baseline

#for i in {1..5};do
 #  CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/train_baseline.yml";
  # python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --index=$i --epochs=25 --gamma=0 --milestones=None --lr=1e-5 --cuda='0' --seed=1;
#done; 

#experiments

#for LAYER in 57 54 51 47 41 38 35 28 25 22 19 15 6; do
 #   for RANK in 1 2 3 4 5 6 7 8 9; do
     #   for i in {1..5}; do
   #        CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tucker-r0.$RANK-$LAYER.yml";
      #     python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.$RANK --index=$i --cuda='0' --seed=1;
     #   done; 
   # done;    
#done; 


#for i in {1..5};do
#    for LAYER in 63 60 57 54 51 47 41 38 35 28 25 22 19 15 6;
 #       do for RANK in 1 2 3 4 5 6 7 8 9;
  #          CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tucker-r0.$RANK-$LAYER.yml";
   #         python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.$RANK --index=$i --cuda='0' --seed=1 --epochs=23;
   #     done; 
   # done;    
#done; 


for i in {1..5};do
    for LAYER in 63 60 57 54 51 47 41 38 35 28 25 22 19 15 6;
        do for RANK in 1 2 3 4 5 6 7 8 9;
            CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml";
            python train.py main --config-path "$CONFIG_PATH" --data_workers=4 --rank=0.$RANK --index=$i --cuda='0' --seed=1;
        done; 
    done;    
done; 
