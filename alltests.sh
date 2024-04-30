#!/bin/bash
#train limits

#resnet18 cifar10
#baseline

#experiments
for i in {1..5};do
    for LAYER in 63 60 57 54 51 47 44 41 38 35 31 28 22 19 15 12 9 6;
        do for RANK in 1 2 3 4 5 6 7 8 9;
            do for FACT in cp tucker;
                CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-$FACT-r0.$RANK-$LAYER.yml";
                python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.$RANK --index=$i --cuda='0';
            done; 
        done; 
    done;    
done; 


for i in {1..5};do
    for LAYER in 63 60 57 54 51 47 44 41 38 35 31 28 22 19 15 12 9 6;
        do for RANK in 1 2 3 4 5 6 7 8 9;
            CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml";
            python train.py main --config-path "$CONFIG_PATH" --data_workers=4 --rank=0.$RANK --index=$i --cuda='0';
        done; 
    done;    
done; 
