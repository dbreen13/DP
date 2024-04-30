#!/bin/bash
#train limits

#resnet18 cifar10
#baseline

#experiments
for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.2-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.2 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.3-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.3 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.4-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.4 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.5-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.5 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.6-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.6 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.7-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.7 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.8-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.8 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.9-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.9 --index=$i --cuda='0';
done; 