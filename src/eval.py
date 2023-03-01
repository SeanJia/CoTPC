import os
import gym
import torch
import numpy as np

from data import MS2Demos
from model import GPTConfig, GPTWithCoT


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
    preds, _ = model(
        states, timesteps, actions=actions, key_state_mask=key_state_mask)
    return preds[:, -1]  # Only output the action predictions.


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


if __name__ == "__main__":
    conf = GPTConfig(
        "Please specify required parameters",  
        # train_data.max_steps
        max_timestep="The max sequence length used during training (for position embedding.)")
    model = GPTWithCoT(
        conf, state_dim='Can use train_data.info()', action_dim='Can use train_data.info()').cuda()

    model_path = "Specify the model ckpt path"
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    # An example eval loop.
    env = gym.make('Some inputs here.')
    s = env.reset()
    state_hist, action_hist, t = [s], [], np.zeros([1])

    for step in range(200):  # Max steps.
        a = predict(model, action_hist, state_hist, t).cpu().numpy()[0]
        
        s, _, _, info = env.step(a)
        s = torch.from_numpy(s).float()[None, :]
        a = a[None, :]
        
        action_hist, state_hist, t = update(model, action_hist, state_hist, a, s, t)
