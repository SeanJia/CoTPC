# I updated part of the environment file in ManiSkill2 so that it allows metrics 
# computation of either reaching an intermediate key state during testing.
# Simply replace the `evaluate` function.
  
"""
Patch the peg_insertion_side env in ManiSkill2 so that it allows additional metrics for 
eval and flags for obtaining key states in training. Please simply replace the `evaluate`
function in (with the correct level of indentation):

https://github.com/haosulab/ManiSkill2/blob/main/mani_skill2/envs/assembly/peg_insertion_side.py
"""

def evaluate(self, **kwargs):
    is_grasped = self.agent.check_grasp(self.peg, max_angle=20)  

    pre_inserted = False
    if is_grasped:
        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_head_wrt_goal_yz_dist = np.linalg.norm(peg_head_wrt_goal.p[1:])
        peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
        peg_wrt_goal_yz_dist = np.linalg.norm(peg_wrt_goal.p[1:])
        if peg_head_wrt_goal_yz_dist < 0.01 and peg_wrt_goal_yz_dist < 0.01:
            pre_inserted = True

    success, peg_head_pos_at_hole = self.has_peg_inserted()
    return dict(
        success=success, 
        pre_inserted=pre_inserted,
        peg_head_pos_at_hole=peg_head_pos_at_hole,
        is_grasped=is_grasped,
    )