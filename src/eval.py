import os
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict

from mani_skill2.utils.io_utils import load_json
import mani_skill2.envs  # Load ManiSkill2 envs.
import torch  # Load pytorch after maniskill2 to avoid some import error.

from model import GPTConfig, GPTWithCoT

from vec_env import get_mp_envs  # Used for parallel evaluation.

try:
    # Use might need this for wandb to work due to protobuf issues.
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    import wandb
    assert wandb.__version__
    USE_WANDB = True
    PROJECT_NAME = 'CoTPC'  # Please specify the project name.
except Exception:
    print('Do not use wandb since it is not found.')
    USE_WANDB = False

# Please specify MODEL_PATH and DATA_PATH (both are base folders) in `path.py`.
from path import MODEL_PATH, DATA_PATH


@torch.no_grad()
def predict(model, action_hist, state_hist, t):
    assert model.model_type in ['s', 's+a', 's+a+cot']  

    timesteps = torch.from_numpy(t)[:, None].cuda()
    if not action_hist:  # The first step.
        actions = None
    else:
        actions = torch.stack(action_hist, 1).float().cuda()
    states = torch.stack(state_hist, 1).float().cuda()

    if 'cot' in model.model_type:
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
    else:
        preds, _ = model(states, timesteps, actions=actions)

    return preds[:, -1]  # Only output the last action predictions.


def update(model, action_hist, state_hist, actions, states, t):
    # A function used to update the state and action history.
    assert model.model_type in ['s', 's+a', 's+a+cot']  

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

    # Hyper-parameters regarding the model.
    parser.add_argument("--model_name", default='', type=str, help="Model name to be loaded.")
    parser.add_argument("--from_ckpt", default=-1, type=int, help="Ckpt of the model to be loaded.")
    
    parser.add_argument("--eval_max_steps", default=200, type=int, help="Max steps allowed in eval.")
    parser.add_argument('--cot_decoder', type=str, default='256', help="Specs of the CoT decoder.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    assert args.model_name, 'Should specify --model_name'
    assert args.from_ckpt > 0, 'Should specify --from_ckpt'

    # Load the model.
    path = os.path.join(MODEL_PATH, f'{args.model_name}/{args.from_ckpt}.pth')
    # Load to cpu first to avoid cuda related errors from ManiSkill2.
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    state_dict_from_ckpt, params = ckpt['model'], ckpt['metadata']
    state_dim = state_dict_from_ckpt['state_encoder.net.0.weight'].shape[1]
    action_dim = state_dict_from_ckpt['action_encoder.net.0.weight'].shape[1]
    max_timestep = state_dict_from_ckpt['global_pos_emb'].shape[1]
    print('Loaded ckpt from:', path)

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
            t_ids = np.random.permutation(
                length_all // 10)[:params['num_traj'] // 10]
            t_ids += i * length_all // 10
            ids.append(t_ids)
        eval_ids = np.concatenate(ids)
    elif args.task == 'PushChair-v1':
        length_all = len(json_data["episodes"])
        ids = []
        for i in range(5):  # Hard-code the 5 data splits for permutation.
            t_ids = np.random.permutation(length_all // 5)[:100]
            t_ids += i * length_all // 5
            ids.append(t_ids)
        eval_ids = np.concatenate(ids)
    else:
        # Only evaluate at most 500 scene configs.
        eval_ids = np.random.permutation(
            len(json_data["episodes"]))[:params['num_traj']][:500]
        
    n_env = 25  # Number of parallel environments.
    assert len(eval_ids) % n_env == 0, f'{len(eval_ids)}'
    envs = get_mp_envs(args.task, n_env, **env_kwargs)

    # Load the ckpt after envs init to avoid cuda related errors from ManiSkill2.
    cot_decoder = params['cot_decoder'] if 'cot_decoder' in params else args.cot_decoder
    conf = GPTConfig(
        params['context_length'], 
        n_layer=params['n_layer'], 
        n_head=params['n_head'], 
        n_embd=params['n_embd'], 
        model_type=params['model_type'], 
        key_states=params['key_states'],  # Rules for the CoT.
        key_state_loss=params['key_state_loss'],  # Layers used for CoT modeling.
        cot_decoder=cot_decoder,
        max_timestep=max_timestep,
    )
    model = GPTWithCoT(conf, state_dim=state_dim, action_dim=action_dim).cuda()
    model.load_state_dict(state_dict_from_ckpt, strict=False) 
    model.eval()

    if USE_WANDB:
        wandb.init(project=PROJECT_NAME, name=f'eval/{args.model_name}', 
                   id=f'wandb_metrics_{args.model_name}', resume='auto')

    output_str, output_dict = '', dict()

    # Seen scene configurations.
    metric_dict = defaultdict(lambda: [[] for _ in range(len(eval_ids))])
    for start_idx in tqdm(range(0, len(eval_ids), n_env)):
        reset_args_list = []
        for i in range(start_idx, min(start_idx + n_env, len(eval_ids))):
            reset_kwargs = json_data["episodes"][eval_ids[i]]['reset_kwargs']
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
                if args.task == 'PickCube-v0':
                    metric_dict['is_grasped'][j].append(info['is_grasped'])
                if args.task == 'StackCube-v0':
                    metric_dict['is_cubaA_grasped'][j].append(info['is_cubaA_grasped'])
                    metric_dict['is_cubeA_on_cubeB'][j].append(info['is_cubeA_on_cubeB'])
                if args.task == 'PegInsertionSide-v0':
                    metric_dict['is_grasped'][j].append(info['is_grasped'])
                    metric_dict['pre_inserted'][j].append(info['pre_inserted'])
                if args.task == 'TurnFaucet-v0':
                    metric_dict['is_contacted'][j].append(info['is_contacted'])
                if args.task == 'PushChair-v1':
                    metric_dict['close_to_target'][j].append(info['chair_close_to_target'])
                    metric_dict['static_at_last'][j].append(
                        info['chair_close_to_target'] and info['chair_static'])
                metric_dict['success'][j].append(info['success'])

    for k, v in metric_dict.items():
        v = np.mean([np.any(vv) for vv in v]) * 100
        output_str += f'{k} {v:.2f}, '
        output_dict[k] = v
    output_str = output_str[:-2]
    print(output_str)

    # Unseen scene configurations.
    # Unseen objects for peg insertion and seen objects otherwise.
    all_reset_kwargs = []
    if args.task == 'TurnFaucet-v0':
        length_all = len(json_data["episodes"])
        ids = []
        for i in range(10):  # Hard-code the 10 data splits for permutation.
            t_ids = np.random.permutation(length_all // 10)
            t_ids = t_ids[params['num_traj']//10:params['num_traj']//10+10]
            t_ids += i * length_all // 10
            ids.append(t_ids)
        eval_ids = np.concatenate(ids)
        for eval_id in eval_ids:
            all_reset_kwargs.append(json_data["episodes"][eval_id]['reset_kwargs'])
    elif args.task == 'PushChair-v1':
        length_all = len(json_data["episodes"])
        ids = []
        for i in range(5):  # Hard-code the 5 data splits for permutation.
            t_ids = np.random.permutation(length_all // 5)
            t_ids = t_ids[params['num_traj']//5:params['num_traj']//5+50]
            t_ids += i * length_all // 5
            ids.append(t_ids)
        eval_ids = np.concatenate(ids)
        for eval_id in eval_ids:
            all_reset_kwargs.append(json_data["episodes"][eval_id]['reset_kwargs'])
    elif args.task == 'PegInsertionSide-v0':
        for i in range(400):
            all_reset_kwargs.append({'seed': i + 2000})
    else:
        for i in range(100):
            all_reset_kwargs.append({'seed': i + 2000})
    metric_dict = defaultdict(lambda: [[] for _ in range(len(all_reset_kwargs))])

    for start_idx in tqdm(range(0, len(all_reset_kwargs), n_env)):
        reset_args_list = []
        for i in range(start_idx, min(start_idx + n_env, len(all_reset_kwargs))):
            reset_args_list.append(all_reset_kwargs[i])

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
                if args.task == 'PickCube-v0':
                    metric_dict['test/is_grasped'][j].append(info['is_grasped'])
                if args.task == 'StackCube-v0':
                    metric_dict['test/is_cubaA_grasped'][j].append(info['is_cubaA_grasped'])
                    metric_dict['test/is_cubeA_on_cubeB'][j].append(info['is_cubeA_on_cubeB'])
                if args.task == 'PegInsertionSide-v0':
                    metric_dict['test/is_grasped'][j].append(info['is_grasped'])
                    metric_dict['test/pre_inserted'][j].append(info['pre_inserted'])
                if args.task == 'TurnFaucet-v0':
                    metric_dict['test/is_contacted'][j].append(info['is_contacted'])
                if args.task == 'PushChair-v1':
                    metric_dict['test/close_to_target'][j].append(info['chair_close_to_target'])
                    metric_dict['test/static_at_last'][j].append(
                        info['chair_close_to_target'] and info['chair_static'])
                metric_dict['test/success'][j].append(info['success'])
       
    output_str = ''
    for k, v in metric_dict.items():
        v = np.mean([np.any(vv) for vv in v]) * 100
        output_str += f'{k} {v:.2f}, '
        output_dict[k] = v
    output_str = output_str[:-2]
    print(output_str)

    # Unseen scene configurations with unseen objects (zero-shot).
    all_reset_kwargs = []
    if args.task == 'TurnFaucet-v0':
        model_ids = [
            5014, 5037, 5053, 5062,
        ]
    elif args.task == 'PushChair-v1':
        model_ids = [
            3003, 3013, 3020,   
        ]
    else:
        model_ids = []
    for model_id in model_ids:
        for i in range(100):
            all_reset_kwargs.append({'seed': i + 2000, 'model_id': str(model_id)})
    metric_dict = defaultdict(lambda: [[] for _ in range(len(all_reset_kwargs))])

    for start_idx in tqdm(range(0, len(all_reset_kwargs), n_env)):
        reset_args_list = []
        for i in range(start_idx, min(start_idx + n_env, len(all_reset_kwargs))):
            reset_args_list.append(all_reset_kwargs[i])

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
                if args.task == 'PushChair-v1':
                    metric_dict['test_h/close_to_target'][j].append(info['chair_close_to_target'])
                    metric_dict['test_h/static_at_last'][j].append(
                        info['chair_close_to_target'] and info['chair_static'])
                if args.task == 'TurnFaucet-v0':
                    metric_dict['test_h/is_contacted'][j].append(info['is_contacted'])
                metric_dict['test_h/success'][j].append(info['success'])
       
    if all_reset_kwargs:
        output_str = ''
        for k, v in metric_dict.items():
            v = np.mean([np.any(vv) for vv in v]) * 100
            output_str += f'{k} {v:.2f}, '
            output_dict[k] = v
        output_str = output_str[:-2]
        print(output_str)

        if USE_WANDB: 
            output_dict['n_iter'] = args.from_ckpt
            wandb.log(output_dict)