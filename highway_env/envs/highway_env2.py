import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs import HighwayEnv
from highway_env.envs.common.action import Action


class HighwayEnv2(HighwayEnv):
    """
    A highway driving environment that include lane change in reward.
    """

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        
        lane_change = action == 0 or action == 2
        
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
            + self.config["lane_change_reward"] * lane_change

        reward = utils.lmap(reward,
                          [self.config["collision_reward"], # + self.config["lane_change_reward"]
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward


register(
    id='highway-v2',
    entry_point='highway_env.envs:HighwayEnv2',
)
