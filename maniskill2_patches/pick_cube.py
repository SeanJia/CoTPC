"""
Patch the pick_cube env in ManiSkill2 so that it allows additional metrics for eval
and flags for obtaining key states in training. Please simply replace the `evaluate`
function in (with the correct level of indentation):

https://github.com/haosulab/ManiSkill2/blob/main/mani_skill2/envs/pick_and_place/pick_cube.py
"""

def evaluate(self, **kwargs):
    is_obj_placed = self.check_obj_placed()
    is_robot_static = self.check_robot_static()
    is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
    return dict(
        is_obj_placed=is_obj_placed,
        is_robot_static=is_robot_static,
        is_grasped=is_grasped,
        success=is_obj_placed and is_robot_static,
    )