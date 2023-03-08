#!/bin/bash

cd ../src && 

python eval.py --num_traj=500 --eval_max_steps=200 \
    --key_states=abc --key_state_loss=0 \
    --from_ckpt=1_800_000 --task=StackCube-v0 \
    --model_name=some_model_name
