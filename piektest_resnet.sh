#!/bin/bash
#train limits

#resnet18 cifar10
#baseline

#experiment1
for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-60.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-63.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-63.60.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-51.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-63.60.54.51.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --cuda='0';
done; 

#experiment 2

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.2-51.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.2 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.30.2-54.51.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.2-54.51.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.2 --index=$i --cuda='0';
done; 