"""
Adapted from template_my_robot_env.py in openai_ros.
"""

import math
import numpy as np
from openai_ros import robot_gazebo_env
import api


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
        self.controllers_list = ['my_robot_controller']

        self.robot_name_space = "my_robot_namespace"

        reset_controls_bool = True
        
        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        
        super(BotEnv, self).__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=reset_controls_bool)
        
        self.bot_api = Orb()
        self.target_api = Target()
        
        self.steps = 0
        self.grid_squares = 0

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
        self.bot_api.randomise_robot_pose()
        self.target_api.randomise_target_pose()
    
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
        if self._is_touching_target():
            return 10000
        
        map_list = self.bot_api.get_latest_slam_map()["data"]
        next_grid_squares = 0
        
        for i in map_list:
            if i == 1:
                next_grid_squares ++
        
        if next_grid_squares > self.grid_squares:
            self.grid_squares = next_grid_squares
            return 50
        else:
            self.grid_squares = next_grid_squares
        
        return -1

    def _set_action(self, action):
        """
        Applies the given action to the simulation.
        """
        self.bot_api.move_robot(action)

    def _get_obs(self):
        camera_data = self.bot_api.get_latest_camera_data()
        camera_array = (np.array(camera_data)).T.flatten()
        
        lidar_data = self.bot_api.get_latest_lidar_data()
        lidar_array = np.array(lidar_data["ranges"])
        
        return np.concatenate((camera_array, lidar_array))

    def _is_done(self, observations):
        """
        Checks if episode done based on observations given.
        """
        self.steps ++
        return self.steps >= 1000 or self._is_touching_target()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    
    def _is_touching_target(self):
        """
        Returns True if the robot is within 5 co-ordinate points of the target.
        """
        bot_pose = self.bot_api.get_ground_truth_robot_pose()
        bot_x = bot_pose["position"]["x"]
        bot_z = bot_pose["position"]["z"]
        
        target_pose = self.target_api.get_ground_truth_target_pose()
        target_x = target_pose["position"]["x"]
        target_z = target_pose["position"]["z"]
        
        x_dist = bot_x - target_x
        z_dist = bot_z - target_z
        dist = math.sqrt((x_dist ** 2) + (z_dist ** 2))
        
        return (dist <= 5)

