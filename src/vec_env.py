import pickle
import cloudpickle
import numpy as np
from multiprocessing import Pipe, Process
import gym

from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

import sapien.core as sapien


def disturb(env, kwargs):
	if 'peg' in kwargs:
		dx, dy, dr = kwargs['peg']
		pose = env.peg.get_pose()
		quat = euler2quat(0, 0, dr)
		env.peg.set_pose(sapien.Pose(
			p=pose.p+[dx,dy,0], q=qmult(quat, pose.q)))
	if 'box' in kwargs:
		dx, dy, dr = kwargs['box']
		pose = env.box.get_pose()
		quat = euler2quat(0, 0, dr)
		env.box.set_pose(sapien.Pose(
			p=pose.p+[dx,dy,0], q=qmult(quat, pose.q)))

def get_mp_envs(env_id, n_env, start_idx=0, **env_kwargs):
    def env_fn(rank):
        def fn():
            env = gym.make(env_id, **env_kwargs)
            return env
        return fn
    return VecEnv([env_fn(i + start_idx) for i in range(n_env)])


class CloudpickleWrapper(object):
	def __init__(self, x):
		self.x = x

	def __getstate__(self):
		return cloudpickle.dumps(self.x)

	def __setstate__(self, ob):
		self.x = pickle.loads(ob)
	
	def __call__(self):
		return self.x()

def worker(remote, parent_remote, env_fn):
	parent_remote.close()
	env = env_fn()
	while True:
		cmd, data = remote.recv()
		
		if cmd == 'step':
			ob, reward, done, info = env.step(data)
			# if done: ob = env.reset()  # We ignore the done signal here.
			remote.send((ob, reward, done, info))
		elif cmd == 'reset':
			ob = env.reset(**data)
			remote.send(ob)
		elif cmd == 'render':
			remote.send(env.render())
		elif cmd == 'close':
			remote.close()
			break
		elif cmd == 'disturb':
			disturb(env, data)
		else:
			raise NameError('NotImplentedError')

class VecEnv():
	def __init__(self, env_fns):
		self.waiting = False
		self.closed = False
		no_envs = len(env_fns)
		self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(no_envs)])
		self.ps = []
		
		for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fns):
			p = Process(target=worker, args=(wrk, rem, CloudpickleWrapper(fn)))
			self.ps.append(p)

		for p in self.ps:
			p.daemon = True
			p.start()

		for remote in self.work_remotes:
			remote.close()

	def step_async(self, actions):
		if self.waiting:
			raise NameError('AlreadySteppingError')
		self.waiting = True
		for remote, action in zip(self.remotes, actions):
			remote.send(('step', action))
	
	def step_wait(self):
		if not self.waiting:
			raise NameError('NotSteppingError')
		self.waiting = False
		results = [remote.recv() for remote in self.remotes]
		obs, rews, dones, infos = zip(*results)
		return np.stack(obs), np.stack(rews), np.stack(dones), infos
	
	def step(self, actions):
		self.step_async(actions)
		return self.step_wait()
	
	def reset(self, kwargs_list):
		for remote, kwargs in zip(self.remotes, kwargs_list):
			remote.send(('reset', kwargs))
		return np.stack([remote.recv() for remote in self.remotes])

	def disturb(self, kwargs_list):
		for remote, kwargs in zip(self.remotes, kwargs_list):
			remote.send(('disturb', kwargs))

	def close(self):
		if self.closed:
			return
		if self.waiting:
			for remote in self.remotes:
				remote.recv()
		for remote in self.remotes:
			remote.send(('close', None))
		for p in self.ps:
			p.join()
		self.closed = True
