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

```cd ~/catkin_ws/```

```catkin_make```

```source ~/catkin_ws/devel/setup.bash```

# launching

```roslaunch orb cafe_world.launch```