#!/bin/bash

# Assume that the *.h5 and *.json are in `../data/rigid_body_envs/TurnFaucet-v0/raw`,
# replay the trajectories with a subset of a total of 10 faucet models.
for s in 5002 5021 5023 5028 5029 5045 5047 5051 5056 5063
do
    python -m mani_skill2.trajectory.replay_trajectory \
        --traj-path ../data/rigid_body_envs/TurnFaucet-v0/raw/$s.h5 \
	    --save-traj --target-control-mode pd_joint_delta_pos \
        --obs-mode state --num-procs 20
done

mv ../data/rigid_body_envs/TurnFaucet-v0/raw/*.state.pd_joint_delta_pos.h5 \
    ../data/rigid_body_envs/TurnFaucet-v0/merged/
mv ../data/rigid_body_envs/TurnFaucet-v0/raw/*.state.pd_joint_delta_pos.json \
    ../data/rigid_body_envs/TurnFaucet-v0/merged/

python -m mani_skill2.trajectory.merge_trajectory \
    -i ../data/rigid_body_envs/TurnFaucet-v0/merged -p *.h5 \
    -o ../data/rigid_body_envs/TurnFaucet-v0/trajectory.state.pd_joint_delta_pos.h5