"""
Adapted from template_my_robot_env.py in openai_ros.
"""

import math
import numpy as np
from openai_ros import robot_gazebo_env
import api
from api import Orb, Target
import rospy

debug = False

class BotEnv(robot_gazebo_env.RobotGazeboEnv):
    """
    Superclass for all Robot environments.
    """

    def __init__(self):
        """
        Initializes a new Robot environment.
        """
        # Variables that we give through the constructor.

        # TODO Not sure what these are for
        # Internal Vars
        # print("START init bot_env")
        self.controllers_list = []

        self.robot_name_space = ""

        reset_controls_bool = False

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv

        super(BotEnv, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=reset_controls_bool,
        )

        self.bot_api = Orb()
        self.target_api = Target()

        self.SKIP = 10
        self.MAX_STEPS = 25
        self.steps = 0
        self.grid_squares = 0
        self.previous_dist = 100000
        self.previous_angle_factor = 100000
        # print("END init bot_env")

    def step(self, action):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done, info
        """

        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """

        self.gazebo.unpauseSim()
        self._set_action(action)
        rate = rospy.Rate(1 / self.SKIP)
        rate.sleep()
        self.gazebo.pauseSim()
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {}
        reward = self._compute_reward(obs, done)
        self.cumulated_episode_reward += reward

        return obs, reward, done, info

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        # TODO Not sure what to do here
        return True

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """
        Sets the Robot in its init pose.
        """
        while True:
            self.previous_dist = 100000
            self.previous_angle_factor = 100000
            self.bot_api.randomise_robot_pose()
            self.target_api.randomise_target_pose()
            
            if not self._is_touching_target():
                break
    
    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        self.steps = 0
        self.grid_squares = 0
        self._set_init_pose()

    def _compute_reward(self, observations, done):
        """
        Calculates the reward to give based on the observations given.
        """
        reward = 0

        # High reward for reaching the target
        if self._is_touching_target():
            reward += 2000
            return reward

        # Reward for map exploration
        map_list = self.bot_api.get_latest_slam_map()["data"]
        next_grid_squares = sum(1 for i in map_list if i == 1)
        if next_grid_squares > self.grid_squares:
            self.grid_squares = next_grid_squares
            reward += 5  # Adjusted reward for exploration

        # Calculate distance to the target
        new_dist = self._distance_to_target()
        distance_change = self.previous_dist - new_dist
        if distance_change > 0.5:
            reward += distance_change * 400  # Reward for moving closer, quicker
        elif distance_change > 0:
            reward += distance_change * 200  # Reward for moving closer
        else:
            reward += distance_change * 60   # Penalty for moving away

        angle_difference = self._calculate_angle_difference()

        if angle_difference < 30 and distance_change > 0:
            reward += distance_change * 100

        angle_difference_rad = math.radians(angle_difference)
        angle_factor = (1 - abs(angle_difference_rad) / (math.pi / 2))

        # Reward/Penalty for looking at the target
        look_at_reward = 0
        angle_factor_change = angle_factor - self.previous_angle_factor
        if angle_factor_change > 0:
            look_at_reward += angle_factor_change * 35  # Reward for looking towards
        else:
            look_at_reward += angle_factor_change * 50   # Penalty for looking away
        reward += look_at_reward

        # Time-based penalty or a small penalty for inaction
        reward -= 2.5

        # Skip first reward
        if self.previous_angle_factor > 1000 or self.previous_dist > 1000:
            reward = 0

        self.previous_dist = new_dist
        self.previous_angle_factor = angle_factor

        print(reward)
        return reward



    def _set_action(self, action):
        """
        Applies the given action to the simulation.
        """
        if action == 0:
            act_string = "f"
        elif action == 1:
            act_string = "b"
        elif action == 2:
            act_string = "cw"
        elif action == 3:
            act_string = "acw"
        else:
            act_string = "stop"

        self.bot_api.move_robot(act_string)

    def _get_obs(self):
        # Process Camera Data
        camera_data = self.bot_api.get_latest_camera_data()
        camera_array = (np.array(camera_data) / 255.0).flatten()

        # Process LIDAR Data
        self.gazebo.unpauseSim()
        lidar_data = self.bot_api.get_latest_lidar_data()
        self.gazebo.pauseSim()
        lidar_array = np.array(lidar_data["ranges"])

        # Concatenate Camera and LIDAR Data
        final = np.concatenate([camera_array, lidar_array])
        #print(final.shape)
        return final


    def _is_done(self, observations):
        """
        Checks if episode done based on observations given.
        """
        self.steps += 1
        return self.steps >= self.MAX_STEPS or self._is_touching_target()
    
    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    def _distance_to_target(self):
        """
        Computes the distance to the target object
        """
        self.gazebo.unpauseSim()
        bot_pose = self.bot_api.get_ground_truth_robot_pose()
        self.gazebo.pauseSim()
        bot_x = bot_pose["position"]["x"]
        bot_y = bot_pose["position"]["y"]

        self.gazebo.unpauseSim()
        target_pose = self.target_api.get_ground_truth_target_pose()
        self.gazebo.pauseSim()
        target_x = target_pose["position"]["x"]
        target_y = target_pose["position"]["y"]

        x_dist = bot_x - target_x
        y_dist = bot_y - target_y
        dist = math.sqrt((x_dist**2) + (y_dist**2))
        if(debug):
            print("Distance: ", dist)
        return dist
    
    def _calculate_angle_difference(self):
        # Get bot's pose and orientation
        self.gazebo.unpauseSim()
        bot_pose = self.bot_api.get_ground_truth_robot_pose()
        self.gazebo.pauseSim()
        bot_x = bot_pose["position"]["x"]
        bot_y = bot_pose["position"]["y"]
        bot_orientation = bot_pose["orientation"]
        roll, pitch, yaw = self._quaternion_to_euler_angles(bot_orientation)

        # Get target's position
        self.gazebo.unpauseSim()
        target_pose = self.target_api.get_ground_truth_target_pose()
        self.gazebo.pauseSim()
        target_x = target_pose["position"]["x"]
        target_y = target_pose["position"]["y"]

        vector_to_target = [target_x - bot_x, target_y - bot_y]
        target_angle_deg = math.degrees(math.atan2(vector_to_target[1], vector_to_target[0]))

        bot_yaw_deg = yaw % 360
        target_angle_deg = target_angle_deg % 360

        if(debug):
            print("Bot Yaw:", bot_yaw_deg)
            print("Target Angle:", target_angle_deg)

        # Calculate the difference in angles
        angle_difference_deg = target_angle_deg - bot_yaw_deg
        angle_difference_deg = (angle_difference_deg + 180) % 360 - 180

        return angle_difference_deg
    
    def _quaternion_to_euler_angles(self, quaternion):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw) with yaw measured from the negative x-axis.
        """
        x, y, z, w = quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Adjust yaw to measure from the negative x-axis
        yaw -= math.pi
        if yaw < -math.pi:
            yaw += 2 * math.pi

        # Convert from radians to degrees
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)

        return roll_deg, pitch_deg, yaw_deg

    def _is_touching_target(self):
        """
        Returns True if the robot is within 2 co-ordinate point of the target.
        """
        # print("RETURNING")
        dst = self._distance_to_target()
        return dst <= 2

    def set_skip(self, newSkip):
        SKIP = newSkip