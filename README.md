# kimm_phri_panda
KIMM pHRI application with Padna Robot Arm

## 1. Prerequisites
### 1.1 Robot controller
    git clone https://github.com/jspark861102/kimm_qpoases.git -b melodic
    git clone https://github.com/jspark861102/kimm_hqp_controller_phri.git -b melodic
    git clone https://github.com/jspark861102/weightedhqp.git -b melodic
    git clone https://github.com/jspark861102/kimm_path_planner.git -b melodic
    git clone https://github.com/jspark861102/kimm_joint_planner_ros_interface.git -b melodic
    git clone https://github.com/jspark861102/kimm_se3_planner_ros_interface.git -b melodic
    git clone https://github.com/jspark861102/kimm_trajectory_smoother.git -b melodic

### 1.2 Robot model and simulator
    git clone https://github.com/jspark861102/franka_ros.git #my used version (0.8.1)
    git clone https://github.com/jspark861102/kimm_robots_description.git -b melodic
    git clone https://github.com/jspark861102/kimm_mujoco_ros.git -b melodic

### 1.3 Unknown obejct parameter estimation
    git clone https://github.com/jspark861102/kimm_object_estimation.git
    git clone https://github.com/jspark861102/kimm_phri_msgs.git

### 1.4 pHRI task
    git clone https://github.com/jspark861102/kimm_phri_panda.git

## 2. Run
### 2.1 Simulation
    # Simulation with PC Monitor
    roslaunch kimm_phri_panda ns0_simulation.launch

    # Simulation with 17inch Notebook
    roslaunch kimm_phri_panda ns0_simulation.launch note_book:=true

### 2.1 Real Robot
    # 1. unknown object parameter estimation
    roslaunch ns0_object_parameter_estimator.launch

    # 2. human-robot collaborative transportation and task
    roslaunch ns0_real_robot.launch

