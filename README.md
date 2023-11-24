# python version
3.8.10

# installation
```source /opt/ros/noetic/setup.bash```

```mkdir -p ~/catkin_ws/src```

```cd ~/catkin_ws/```

```catkin_make```

```source ~/catkin_ws/devel/setup.bash```

```sudo apt-get install gazebo11```

```sudo apt-get install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control```

```sudo apt-get install ros-noetic-xacro```

```sudo apt-get install ros-noetic-pr2-teleop```

```sudo apt-get install ros-noetic-gmapping```

```sudo apt-get install ros-noetic-explore-lite```

```sudo apt-get install ros-noetic-navigation #not sure this is necessary```

```sudo apt-get install ros-noetic-move-base```

```cd ~/catkin_ws/src```

```git clone git@github.com:IRUOB/socspioneer.git```

```git clone git@github.com:ivegotbigideas/ORB.git```

```mv ~/catkin_ws/src/ORB/ ~/catkin_ws/src/orb/```

```mkdir ~/catkin_ws/src/orb/worlds```

```cp /usr/share/gazebo-11/worlds/cafe.world /home/little/catkin_ws/src/orb/worlds/```

# install openai_ros

```cd ~/catkin_ws/src```

```git clone https://github.com/edowson/openai_ros.git```

```cd ~/catkin_ws/src/openai_ros/openai_ros```

edit line in ```package.xml``` from ```<build_depend>python-catkin-pkg</build_depend>``` to ```<build_depend>python3-catkin-pkg</build_depend>```

# make workspace

```cd ~/catkin_ws/```

```catkin_make```

```source ~/catkin_ws/devel/setup.bash```

# install dependencies

```rosdep install openai_ros```

# install pytorch

```pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118```

```pip install tensordict==0.2.0```

```pip install torchrl==0.2.0```

```pip install gymnasium```

```pip install gym```

# launching

```roslaunch orb cafe_world.launch```

