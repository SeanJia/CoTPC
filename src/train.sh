#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name=test_stackcube \
    --num_traj=500 --n_iters=300000 \
    --context_length=60 --model_type=s+a+cot \
    --task=StackCube-v0 --key_state_coeff=1.0 \
    --key_state_loss=0 --key_states=abc
