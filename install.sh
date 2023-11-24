#! /bin/sh.
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
source ~/catkin_ws/devel/setup.bash
sudo apt-get install gazebo11
sudo apt-get install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
sudo apt-get install ros-noetic-xacro
sudo apt-get install ros-noetic-pr2-teleop
sudo apt-get install ros-noetic-gmapping
sudo apt-get install ros-noetic-explore-lite
sudo apt-get install ros-noetic-navigation
sudo apt-get install ros-noetic-move-base
cd ~/catkin_ws/src
git clone git@github.com:IRUOB/socspioneer.git
git clone git@github.com:ivegotbigideas/ORB.git
mv ~/catkin_ws/src/ORB/ ~/catkin_ws/src/orb/
mkdir ~/catkin_ws/src/orb/worlds
cp /usr/share/gazebo-11/worlds/cafe.world /home/little/catkin_ws/src/orb/worlds/
git clone https://github.com/edowson/openai_ros.git
sed -i -e 's/python-catkin-pkg/python3-catkin-pkg/g' openai_ros/openai_ros/package.xml
cd ~/catkin_ws/
catkin_make
source ~/catkin_ws/devel/setup.bash
rosdep install openai_ros
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensordict==0.2.0
pip install torchrl==0.2.0

