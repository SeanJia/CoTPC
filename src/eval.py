import os
import gym
import torch
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict

from mani_skill2.utils.io_utils import load_json

import mani_skill2.envs  # To load ManiSkill2 envs.
from model_bak import GPTConfig, GPTWithCoT

from vec_env import get_mp_envs  # Used for parallel evaluation.

# Please specify the model and data path (base folder).
MODEL_PATH = '/home/zjia/Research/inter_seq/CoTPC/models'
DATA_PATH = '/home/zjia/Research/inter_seq/data/rigid_body_envs'  


@torch.no_grad()
def predict(model, action_hist, state_hist, t):
    # Please modify this function for model_type other than `s+a+cot`.
    assert model.model_type == 's+a+cot'  

    timesteps = torch.from_numpy(t)[:, None].cuda()
    if not action_hist:  # The first step.
        actions = None
    else:
        actions = torch.stack(action_hist, 1).float().cuda()
    states = torch.stack(state_hist, 1).float().cuda()

    # T is the max sequence size; S is the current number of steps.
    B, T = states.shape[0], model.block_size + model.len_key_states
    n_head, S = model.config.n_head, states.shape[1] - 1  # Exclude the init state.

    # Masks for the all-to-all key state query tokens in attention layers.
    # The built-in masks for causal (auto-regressive) tokens are in `model.py`.
    key_state_mask = torch.zeros([B, n_head, T, T], dtype=bool)
    m1 = torch.arange(0, T).repeat(B, 1)
    m2 = torch.ones([B, 1]) * (S * 2 + model.len_key_states)
    m3 = m1 > m2  # Tokens in the future are masked out.
    m3 = m3[:, None, None, :].repeat(1, n_head, model.len_key_states, 1)
    key_state_mask[:, :, :model.len_key_states, :] = m3
    key_state_mask = key_state_mask.cuda()
    preds, _ = model(
        states, timesteps, actions=actions, key_state_mask=key_state_mask)
    return preds[:, -1]  # Only output the last action predictions.


def update(model, action_hist, state_hist, actions, states, t):
    # A function used to update the state and action history.
    # Please change this function for model_type other than `s+a+cot`.
    assert model.model_type == 's+a+cot'  

    actions = torch.from_numpy(actions)
    if len(state_hist) == model.block_size // 2:  # The context buffer is full.
        assert len(action_hist) == model.block_size // 2 - 1
        state_hist = state_hist[1:] + [states]
        action_hist = action_hist[1:] + [actions]
        t += 1
    else:
        state_hist.append(states)
        action_hist.append(actions)
    return action_hist, state_hist, t


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyper-parameters regarding the demo dataset (used to gather eval_ids)
    parser.add_argument('--task', type=str, default='PickCube-v0', help="Task (env-id) in ManiSkill2.")
    parser.add_argument('--control_mode', type=str, default='pd_joint_delta_pos', 
                        help="Control mode used in envs from ManiSkill2.")
    parser.add_argument('--obs_mode', type=str, default='state', 
                        help="State mode used in envs from ManiSkill2.")
    parser.add_argument("--seed", default=0, type=int,help="Random seed for data spliting.")
    parser.add_argument("--num_traj", default=-1, type=int, help="Number of training trajectories.")

    # Hyper-parameters regarding the model.
    parser.add_argument('--context_length', type=int, default=60, 
                        help="Context size of CoTPC.")
    parser.add_argument("--n_layer", default=4, type=int, help="Number of attention layers.")
    parser.add_argument("--n_head", default=8, type=int, help="Number of attention heads.")
    parser.add_argument("--n_embd", default=128, type=int, help="Hidden feature dimension.")
    parser.add_argument('--model_type', type=str, default='s+a+cot', 
                        help="Model type for the CoTPC model (see GPTConfig).")
    parser.add_argument('--key_states', type=str, default='a', 
                        help="Which key states to use (see GPTConfig for the spec. format).")
    parser.add_argument("--key_state_loss", default='', type=str, 
                        help="Features out of what attention layers to use for key state prediction " +
                        "losses (see GPTConfig for the spec. format).")
    parser.add_argument("--max_steps", default=-1, type=int, 
                        help="Max steps of seqences in training data (for loading the pos enbedding).")
    parser.add_argument("--model_name", default='', type=str, help="Model name to be loaded.")
    parser.add_argument("--from_ckpt", default=-1, type=int, help="Ckpt of the model to be loaded.")
    parser.add_argument("--state_dim", default=-1, type=int, help="Dim of the state space.")
    parser.add_argument("--action_dim", default=8, type=int, help="Dim of the action space.")
    
    parser.add_argument("--eval_max_steps", default=200, type=int, help="Max steps allowed in eval.")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    assert args.max_steps > 0, 'Should specify --max_steps'
    assert args.model_name, 'Should specify --model_name'
    assert args.from_ckpt > 0, 'Should specify --from_ckpt'
    assert args.state_dim > 0, 'Should specify --state_dim'

    # Load the model.
    conf = GPTConfig(
        args.context_length, 
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd, 
        model_type=args.model_type, 
        key_states=args.key_states,
        key_state_loss=args.key_state_loss,
        max_timestep=args.max_steps,
    )
    model = GPTWithCoT(conf, state_dim=args.state_dim, action_dim=args.action_dim).cuda()
    path = os.path.join(MODEL_PATH, f'{args.model_name}/{args.from_ckpt}.pth')
    print('Loaded ckpt from:', path)  
    model.load_state_dict(torch.load(path), strict=False) 
    model.eval()

    # Load demos to fetch the env. seeds used in training.
    json_path = os.path.join(
        DATA_PATH, f'{args.task}/trajectory.{args.obs_mode}.{args.control_mode}.json')
    json_data = load_json(json_path)
    env_kwargs = json_data["env_info"]["env_kwargs"]
    env_kwargs["obs_mode"] = args.obs_mode
    env_kwargs["control_mode"] = args.control_mode
    np.random.seed(args.seed)
    if args.task == 'TurnFaucet-v0':
        length_all = len(json_data["episodes"])
        ids = []
        for i in range(10):  # Hard-code the 10 data splits for permutation.
            t_ids = np.random.permutation(length_all//10)[:args.num_traj//10]
            t_ids += i*length_all//10
            ids.append(t_ids)
        eval_ids = np.concatenate(ids)
    else:
        eval_ids = np.random.permutation(len(json_data["episodes"]))[:args.num_traj]

    # Number of parallel environments.
    n_env = 25
    assert len(eval_ids) % n_env == 0, f'{len(eval_ids)}'
    envs = get_mp_envs(args.task, n_env, **env_kwargs)

    metric_dict = defaultdict(lambda: [[] for _ in range(len(eval_ids))])

    for start_idx in tqdm(range(0, len(eval_ids), n_env)):
        reset_args_list = []
        for i in range(start_idx, min(start_idx + n_env, len(eval_ids))):
            reset_kwargs = {'seed': json_data["episodes"][eval_ids[i]]["episode_seed"]}
            reset_args_list.append(reset_kwargs)

        s = torch.from_numpy(envs.reset(reset_args_list)).float()
        state_hist, action_hist, t = [s], [], np.zeros([n_env])

        for step in range(args.eval_max_steps):
            a = predict(model, action_hist, state_hist, t).cpu().numpy()

            s, _, _, infos = envs.step(a)
            s = torch.from_numpy(s).float()
            
            action_hist, state_hist, t = update(
                model, action_hist, state_hist, a, s, t)
            
            # Update metrics.
            for i, info in enumerate(infos):
                j = start_idx + i   
                # You might want to use these additional metrics.         
                # if args.task == 'PickCube-v0':
                #     metric_dict['is_grasped'][j].append(info['is_grasped'])
                # if args.task == 'StackCube-v0':
                #     metric_dict['is_cubaA_grasped'][j].append(info['is_cubaA_grasped'])
                #     metric_dict['is_cubeA_on_cubeB'][j].append(info['is_cubeA_on_cubeB'])
                # if args.task == 'PegInsertionSide-v0':
                #     metric_dict['is_grasped'][j].append(info['is_grasped'])
                #     metric_dict['pre_inserted'][j].append(info['pre_inserted'])
                # if args.task == 'TurnFaucet-v0':
                #     metric_dict['is_contacted'][j].append(info['is_contacted'])
                metric_dict['success'][j].append(info['success'])
            
    output_str = ''
    for k, v in metric_dict.items():
        v = np.mean([np.any(vv) for vv in v]) * 100
        output_str += f'{k} {v:.2f}, '
    output_str = output_str[:-2]
    print(output_str)

    # Example eval loop with a single env (not the paralleled VecEnv).
    # env = gym.make('Some inputs here.')
    # s = env.reset()
    # state_hist, action_hist, t = [s], [], np.zeros([1])

    # for step in range(args.eval_max_steps):
    #     a = predict(model, action_hist, state_hist, t).cpu().numpy()[0]
        
    #     s, _, _, info = env.step(a)
    #     s = torch.from_numpy(s).float()[None, :]
    #     a = a[None, :]
        
    #     action_hist, state_hist, t = update(model, action_hist, state_hist, a, s, t)
