#!/bin/bash
#CUDA_VISIBLE_DEVICES=0;nohup bash -c "(python run_ortho.py CIFAR 0.0001 0 0 > cifar_0001_0_0.out) &> cpython_0001_0_0.out" &
#CUDA_VISIBLE_DEVICES=1;nohup bash -c "(python run_ortho.py CIFAR 0.0005 0 1 > cifar_0005_0_1.out) &> cpython_0005_0_1.out" &
#CUDA_VISIBLE_DEVICES=2;nohup bash -c "(python run_ortho.py CIFAR 0.001 0 2 > cifar_001_0_2.out) &> cpython_001_0_2.out" &
#CUDA_VISIBLE_DEVICES=3;nohup bash -c "(python run_ortho.py CIFAR 0.0001 1 3 > cifar_0001_1_3.out) &> cpython_0001_1_3.out" &
#CUDA_VISIBLE_DEVICES=4;nohup bash -c "(python run_ortho.py CIFAR 0.0005 1 4 > cifar_0005_1_4.out) &> cpython_0005_1_4.out" &
#CUDA_VISIBLE_DEVICES=5;nohup bash -c "(python run_ortho.py CIFAR 0.001 1 5 > cifar_001_1_5.out) &> cpython_001_1_5.out" &
#CUDA_VISIBLE_DEVICES=6;nohup bash -c "(python run_ortho.py SVHN 0.0001 0 6 > svhn_0001_0_0.out) &> spython_0001_0_0.out" &
#CUDA_VISIBLE_DEVICES=3;nohup bash -c "(python run_ortho.py SVHN 0.0005 0 3 > svhn_0005_0_1.out) &> spython_0005_0_1.out" &
#CUDA_VISIBLE_DEVICES=0;nohup bash -c "(python run_ortho.py SVHN 0.001 0 0 > svhn_001_0_2.out) &> spython_001_0_2.out" &
#CUDA_VISIBLE_DEVICES=1;nohup bash -c "(python run_ortho.py SVHN 0.0001 1 1 > svhn_0001_1_3.out) &> spython_0001_1_3.out" &
#CUDA_VISIBLE_DEVICES=2;nohup bash -c "(python run_ortho.py SVHN 0.0005 1 2 > svhn_0005_1_4.out) &> spython_0005_1_4.out" &
#CUDA_VISIBLE_DEVICES=7;nohup bash -c "(python run_ortho.py SVHN 0.001 1 7 > svhn_001_1_5.out) &> spython_001_1_5.out" &


#export CUDA_VISIBLE_DEVICES=5;nohup bash -c "(python run_ortho.py IMAGE 0.0001 2 0 > cifar_0001_0_0.out) &> cpython_0001_0_0.out" &
#sleep 10s
#export CUDA_VISIBLE_DEVICES=6;nohup bash -c "(python run_ortho.py IMAGE 0.0005 2 0 > cifar_0005_0_1.out) &> cpython_0005_0_1.out" &
#sleep 10s
#export CUDA_VISIBLE_DEVICES=2;nohup bash -c "(python run_ortho.py IMAGE 0.001 2 0 > cifar_001_0_2.out) &> cpython_001_0_2.out" &
#sleep 10s
#export CUDA_VISIBLE_DEVICES=3;nohup bash -c "(python run_ortho.py IMAGE 0.0001 1 0 > cifar_0001_0_0.out) &> cpython_0001_0_0.out" &
#sleep 10s
#export CUDA_VISIBLE_DEVICES=1;nohup bash -c "(python run_ortho.py IMAGE 0.0005 1 0 > cifar_0005_0_1.out) &> cpython_0005_0_1.out" &
#sleep 10s
#export CUDA_VISIBLE_DEVICES=4;nohup bash -c "(python run_ortho.py IMAGE 0.001 1 0 > cifar_001_0_2.out) &> cpython_001_0_2.out" &


export CUDA_VISIBLE_DEVICES=2;nohup bash -c "(python run_ortho.py CIFAR100 0.0005 2 0 > cifar100_0005_0_2.out) &> cpython100_0005_0_2.out" &
sleep 10s
#export CUDA_VISIBLE_DEVICES=2;nohup bash -c "(python run_ortho.py CIFAR100 0.0005 2 0 > cifar100_0005_1_2.out) &> cpython100_0005_1_2.out" &
#sleep 10s
#export CUDA_VISIBLE_DEVICES=1;nohup bash -c "(python run_ortho.py CIFAR100 0 0.0005 1 0 > cifar100_0005_0_1.out) &> cpython100_0005_0_1.out" &
#sleep 10s
#export CUDA_VISIBLE_DEVICES=0;nohup bash -c "(python run_ortho.py CIFAR100 0.001 2 0 > cifar_001_0_2.out) &> cpython_001_0_2.out" &
#sleep 10s
#export CUDA_VISIBLE_DEVICES=1;nohup bash -c "(python run_ortho.py CIFAR100 0.001 2 0 > cifar_001_0_2.out) &> cpython_001_0_2.out" &









