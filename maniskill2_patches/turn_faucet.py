"""
Patch the turn_faucet env in ManiSkill2 so that it allows additional metrics for eval
and flags for obtaining key states in training. Please replace the `evaluate` function
in (with the correct level of indentation):

https://github.com/haosulab/ManiSkill2/blob/main/mani_skill2/envs/misc/turn_faucet.py

Moroever, replace the `_get_obs_extra` function to customize the state representation
so that it is easier to distinguish among different faucet models.

Note: 
The 10 faucet models (a simpler subset of all faucets in ManiSkill2) we use in the 
CoTPC paper have the ids: 5002,5021,5023,5028,5029,5045,5047,5051,5056,5063.
"""

def _get_curr_target_link_pos(self):
    """
    Access the current pose of the target link (i.e., the handle of the faucet).
    """
    cmass_pose = self.target_link.pose * self.target_link.cmass_local_pose
    return cmass_pose.p

def _get_obs_extra(self):
    obs = OrderedDict(
        tcp_pose=vectorize_pose(self.tcp.pose),
        target_angle_diff=self.target_angle_diff,
        target_joint_axis=self.target_joint_axis,
        target_link_pos=self.target_link_pos,
        curr_target_link_pos=self._get_curr_target_link_pos(),  # Added code.
    )
    if self._obs_mode in ["state", "state_dict"]:
        angle_dist = self.target_angle - self.current_angle
        obs["angle_dist"] = angle_dist
    return obs

def evaluate(self, **kwargs):
    is_contacted = any(self.agent.check_contact_fingers(self.target_link))
    angle_dist = self.target_angle - self.current_angle
    return dict(
        success=angle_dist < 0, 
        angle_dist=angle_dist,
        is_contacted=is_contacted)