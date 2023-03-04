"""
Patch the record utility in ManiSkill2 so that it records additional metrics for eval
and flags for obtaining key states in training. Please patch the `flush_trajectory`
function in (with the correct level of indentation):

https://github.com/haosulab/ManiSkill2/blob/main/mani_skill2/utils/wrappers/record.py
"""

def flush_trajectory(self, **args):

    # some code here ...
    
    ########################### ADDED CODE #############################
    # Append info (boolean flags) to the recorded trajectories.
    # This tells you what info to store in the trajs.
    info_bool_keys = []
    for k, v in self._episode_data[-1]['info'].items():
        if type(v).__module__ == np.__name__ and v.dtype == 'bool':
            info_bool_keys.append(k)
        elif isinstance(v, bool):
            info_bool_keys.append(k)

    # This info only appears in some trajs.
    if 'TimeLimit.truncated' in info_bool_keys:
        info_bool_keys.remove('TimeLimit.truncated')
    ####################################################################

    if len(self._episode_data) == 1:
        # some code here ...

        ########################### ADDED CODE #############################
        infos_bool = {k: np.empty(shape=(0,), dtype=bool) for k in info_bool_keys}
        ####################################################################
    else:
        # some code here ...

        ########################### ADDED CODE #############################
        infos_bool = {k: [] for k in info_bool_keys}
        for x in self._episode_data[1:]:
            for k in info_bool_keys:
                infos_bool[k].append(x['info'][k])
        for k in infos_bool:
            infos_bool[k] = np.stack(infos_bool[k])
        ####################################################################

    # some code here ...

    ########################### ADDED CODE #############################
    # Dump the additional entries to the demo trajectories.
    rewards = np.array([x['r'] for x in self._episode_data[1:]], dtype=np.float32)
    group.create_dataset("rewards", data=rewards, dtype=np.float32)
    for k, v in infos_bool.items():
        group.create_dataset(f"infos/{k}", data=v, dtype=bool)
    ####################################################################

    # Handle JSON
    # some code here ...