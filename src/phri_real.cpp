
#include <kimm_phri_panda/phri_real.h>
#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka/robot_state.h>

namespace kimm_franka_controllers
{

bool BasicFrankaController::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle)
{

  node_handle.getParam("/robot_group", group_name_);
  
  ctrl_type_sub_ = node_handle.subscribe("/" + group_name_ + "/real_robot/ctrl_type", 1, &BasicFrankaController::ctrltypeCallback, this);
  mob_subs_ = node_handle.subscribe("/" + group_name_ + "/real_robot/mob_type", 1, &BasicFrankaController::mobtypeCallback, this);
  
  torque_state_pub_ = node_handle.advertise<mujoco_ros_msgs::JointSet>("/" + group_name_ + "/real_robot/joint_set", 5);
  joint_state_pub_ = node_handle.advertise<sensor_msgs::JointState>("/" + group_name_ + "/real_robot/joint_states", 5);
  time_pub_ = node_handle.advertise<std_msgs::Float32>("/" + group_name_ + "/time", 1);

  ee_state_pub_ = node_handle.advertise<geometry_msgs::Transform>("/" + group_name_ + "/real_robot/ee_state", 5);
  ee_state_msg_ = geometry_msgs::Transform();  

  // panda load pub
  panda_load_parameter_pub_ = node_handle.advertise<kimm_phri_msgs::ObjectParameter>("/" + group_name_ + "/real_robot/panda_load_parameter", 5);

  // set load
  setload_client = node_handle.serviceClient<franka_msgs::SetLoad>("/franka_control/set_load");  

  isgrasp_ = true;    // gripper is initially closed during object parameter estimation procedure
  
  gripper_ac_.waitForServer();
  gripper_grasp_ac_.waitForServer();

  std::vector<std::string> joint_names;
  std::string arm_id;  
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("Could not read parameter arm_id");
    return false;
  }
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "Invalid or no joint_names parameters provided, aborting "
        "controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM("Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "Exception getting model handle from interface: " << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM("Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "Exception getting state handle from interface: " << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM("Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM("Exception getting joint handles: " << ex.what());
      return false;
    }
  }  

  //set external load --------------------------------------------------------------//
  std::vector<double> obj_com, obj_inertia;
  if (node_handle.getParam("/object_parameter/mass", obj_mass_)) 
  {
    node_handle.getParam("/object_parameter/center_of_mass", obj_com);
    node_handle.getParam("/object_parameter/inertia", obj_inertia);
    
    obj_com_ = Eigen::Map<Vector3d>(obj_com.data());
    obj_inertia_ = Eigen::Map<Vector9d>(obj_inertia.data());

    setload_srv.request.mass = obj_mass_;
    setload_srv.request.F_x_center_load[0] = obj_com_[0];
    setload_srv.request.F_x_center_load[1] = obj_com_[1];
    setload_srv.request.F_x_center_load[2] = obj_com_[2];  
    setload_srv.request.load_inertia[0] = 0.0;  
    setload_srv.request.load_inertia[1] = 0.0;  
    setload_srv.request.load_inertia[2] = 0.0;  
    setload_srv.request.load_inertia[3] = 0.0;  
    setload_srv.request.load_inertia[4] = 0.0;  
    setload_srv.request.load_inertia[5] = 0.0;  
    setload_srv.request.load_inertia[6] = 0.0;  
    setload_srv.request.load_inertia[7] = 0.0;  
    setload_srv.request.load_inertia[8] = 0.0;  

    
    // setload_client.call(setload_srv);    
    if(setload_client.call(setload_srv))
    {
      ROS_INFO("set load succeed");
    }
    else
    {
      ROS_INFO("Failed to call service");
    }
  }
  else 
  {
    ROS_INFO("object parameter is not loaded.");

  }

  //keyboard event
  mode_change_thread_ = std::thread(&BasicFrankaController::modeChangeReaderProc, this);

  // ctrl_ = new RobotController::FrankaWrapper(group_name_, false, node_handle);
  ctrl_ = new RobotController::FrankaWrapper(group_name_, false, node_handle, 1);
  ctrl_->initialize();  

  cout << " " << endl;
  cout << "home position" << endl;
  cout << " " << endl;
  
  return true;
}

void BasicFrankaController::starting(const ros::Time& time) {  
  time_ = 0.;
  dt_ = 0.001;

  mob_type_ = 0;

  robot_command_msg_.torque.resize(7); // 7 (franka) 
  robot_state_msg_.position.resize(9); // 7 (franka) + 2 (gripper)
  robot_state_msg_.velocity.resize(9); // 7 (franka) + 2 (gripper)  

  dq_filtered_.setZero();
  f_filtered_.setZero();     

  //franka state_handle to check load parameter -------------------------------//
  franka::RobotState robot_state = state_handle_->getRobotState(); 

  double m_load(robot_state.m_load);
  m_load_ = m_load; //external load's mass.   

  Eigen::Map<Eigen::Matrix<double, 3, 1>> F_x_Cload(robot_state.F_x_Cload.data());
  F_x_Cload_ = F_x_Cload; //external load's com   

  Eigen::Map<Eigen::Matrix<double, 9, 1>> I_load(robot_state.I_load.data());
  I_load_ = I_load; //external load's inertia
  
  ROS_INFO_STREAM("m_load_ :" << m_load_);
  ROS_INFO_STREAM("F_x_Cload_ :" << "[" << F_x_Cload_[0] << "  " << F_x_Cload_[1] << "  " << F_x_Cload_[2] << "]");
  ROS_INFO_STREAM("I_load_ :" << I_load_[0] << "  " << I_load_[1] << "  " << I_load_[2] << "  " << I_load_[3] << "  " << I_load_[4] << "  " << I_load_[5] << "  " << I_load_[6] << "  " << I_load_[7] << "  " << I_load_[8]);
}

void BasicFrankaController::update(const ros::Time& time, const ros::Duration& period) {
  
  //update franka variables----------------------------------------------------------------------//  
  //franka model_handle -------------------------------//
  std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  std::array<double, 7> gravity_array = model_handle_->getGravity();
  std::array<double, 49> massmatrix_array = model_handle_->getMass();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();  

  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  robot_J_ = jacobian; //Gets the 6x7 Jacobian for the given joint relative to the base frame
  
  Eigen::Map<Vector7d> gravity(gravity_array.data());
  robot_g_ = gravity; //Calculates the gravity vector [Nm]

  Eigen::Map<Matrix7d> mass_matrix(massmatrix_array.data());
  robot_mass_ = mass_matrix; //Calculates the 7x7 mass matrix [kg*m^2]

  Eigen::Map<Vector7d> non_linear(coriolis_array.data());
  robot_nle_ = non_linear; //Calculates the Coriolis force vector (state-space equation) [Nm]

  // can be used
  //model_handle_->getpose(); //Gets the 4x4 pose matrix for the given frame in base frame
  //model_handle_->getBodyJacobian(); //Gets the 6x7 Jacobian for the given frame, relative to that frame
  
  //franka state_handle -------------------------------//
  franka::RobotState robot_state = state_handle_->getRobotState();

  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J(robot_state.tau_J.data());
  robot_tau_ = tau_J; //Measured link-side joint torque sensor signals [Nm]

  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(robot_state.tau_J_d.data());
  robot_tau_d_ = tau_J_d; //Desired link-side joint torque sensor signals without gravity [Nm]    
  
  Eigen::Map<Vector7d> franka_q(robot_state.q.data());
  franka_q_ = franka_q; //Measured joint position [rad] 

  Eigen::Map<Vector7d> franka_dq(robot_state.dq.data());
  franka_dq_ = franka_dq; //Measured joint velocity [rad/s]  

  Eigen::Map<Vector7d> franka_dq_d(robot_state.dq_d.data());
  franka_dq_d_ = franka_dq_d; //Desired joint velocity [rad/s]  
  
  Eigen::Map<Eigen::Matrix<double, 6, 1>> force_franka(robot_state.O_F_ext_hat_K.data());
  f_ = force_franka; //Estimated external wrench (force, torque) acting on stiffness frame, expressed relative to the base frame.   

  double m_load(robot_state.m_load);
  m_load_ = m_load; //external load's mass.   

  Eigen::Map<Eigen::Matrix<double, 3, 1>> F_x_Cload(robot_state.F_x_Cload.data());
  F_x_Cload_ = F_x_Cload; //external load's com   

  Eigen::Map<Eigen::Matrix<double, 9, 1>> I_load(robot_state.I_load.data());
  I_load_ = I_load; //external load's inertia

  // can be used
  //robot_state.tau_ext_hat_filtered.data(); //External torque, filtered. [Nm]

  //franka End-Effector Frame -----------------------------------------------------------------//  
  // 1. Nominal end effector frame NE : The nominal end effector frame is configure outsuide of libfranka and connot changed here.
  // 2. End effector frame EE : By default, the end effector frame EE is the same as the nominal end effector frame NE (i.e, the transformation between NE and EE is the identity transformation)
  //                            With Robot::setEE, a custom transformation matrix can be set
  // 3. Stiffness frame K : The stiffness frame is used for Cartesian impedance control, and for measuring and applying forces. I can be set with Robot::setK
  
  //filtering ---------------------------------------------------------------------------------//  
  dq_filtered_      = lowpassFilter( dt_,  franka_dq,  dq_filtered_,       20.0); //in Hz, Vector7d
  f_filtered_       = lowpassFilter( dt_,  f_,         f_filtered_,        20.0); //in Hz, Vector6d  
  
  // thread for franka state update to HQP -----------------------------------------------------//
  if (calculation_mutex_.try_lock())
  {
    calculation_mutex_.unlock();
    if (async_calculation_thread_.joinable())
      async_calculation_thread_.join();

    //asyncCalculationProc -->  ctrl_->franka_update(franka_q_, dq_filtered_);
    async_calculation_thread_ = std::thread(&BasicFrankaController::asyncCalculationProc, this);
  }

  ros::Rate r(30000);
  for (int i = 0; i < 7; i++)
  {
    r.sleep();
    if (calculation_mutex_.try_lock())
    {
      calculation_mutex_.unlock();
      if (async_calculation_thread_.joinable())
        async_calculation_thread_.join();
      break;
    }
  }
  
  // compute HQP controller --------------------------------------------------------------------//
  //obtain from panda_hqp --------------------//
  ctrl_->compute(time_);  
  ctrl_->franka_output(franka_qacc_); 

  //for object estimation--------------------//
  // ctrl_->ddq(franka_ddq_);               //ddq is obtained from pinocchio ABA algorithm  
  // ctrl_->mass(robot_mass_);              //use franka api mass, not pinocchio mass
  robot_mass_(4, 4) *= 6.0;                 //practical term? for gain tuining?
  robot_mass_(5, 5) *= 6.0;                 //practical term? for gain tuining?
  robot_mass_(6, 6) *= 10.0;                //practical term? for gain tuining?
  franka_torque_ = robot_mass_ * franka_qacc_ + robot_nle_;  

  MatrixXd Kd(7, 7); // this is practical term
  Kd.setIdentity();
  Kd = 2.0 * sqrt(5.0) * Kd;
  Kd(5, 5) = 0.2;
  Kd(4, 4) = 0.2;
  Kd(6, 6) = 0.2; 
  franka_torque_ -= Kd * dq_filtered_;  
  franka_torque_ << this->saturateTorqueRate(franka_torque_, robot_tau_d_);

  //for admittance control & robust control--------------------//
  if (ctrl_->ctrltype() != 0){
    if (mob_type_ == 1){
      franka_torque_ -= robot_J_.transpose().col(0) * f_filtered_(0); //admittance
      franka_torque_ -= robot_J_.transpose().col(1) * f_filtered_(1); //admittance
      franka_torque_ += robot_J_.transpose().col(2) * f_filtered_(2); //robust
    }
    else if (mob_type_ == 2){
      franka_torque_ += robot_J_.transpose().col(0) * f_filtered_(0); //robust
      franka_torque_ += robot_J_.transpose().col(1) * f_filtered_(1); //robust
      franka_torque_ += robot_J_.transpose().col(2) * f_filtered_(2); //robust

      // franka_torque_ += robot_J_.transpose().col(3) * f_filtered_(3);  //don't do this code , dangerous!
      // franka_torque_ += robot_J_.transpose().col(4) * f_filtered_(4);  //don't do this code , dangerous!
      // franka_torque_ += robot_J_.transpose().col(5) * f_filtered_(5);  //don't do this code , dangerous!
    }
  }

  //send control input to franka--------------------//
  for (int i = 0; i < 7; i++)
    joint_handles_[i].setCommand(franka_torque_(i));  

  //Publish ------------------------------------------------------------------------//
  time_ += dt_;
  time_msg_.data = time_;
  time_pub_.publish(time_msg_);    

  this->getEEState(); //just update ee_state_msg_ by pinocchio and publish it
  
  for (int i=0; i<7; i++){  // just update franka state(franka_q_, dq_filtered_) 
      robot_state_msg_.position[i] = franka_q(i);
      robot_state_msg_.velocity[i] = dq_filtered_(i);
  }
  joint_state_pub_.publish(robot_state_msg_);
    
  this->setFrankaCommand(); //just update robot_command_msg_ by franka_torque_ 
  torque_state_pub_.publish(robot_command_msg_);

  //pub panda load parameter -----------------------------------------------------//
  this->panda_load_parameter_pub();  

  //Debug ------------------------------------------------------------------------//
  if (print_rate_trigger_())
  {
    // ROS_INFO("--------------------------------------------------");
    // ROS_INFO_STREAM("robot_mass_ :" << robot_mass_);
    // ROS_INFO_STREAM("m_load_ :" << m_load_);
    // ROS_INFO_STREAM("odom_lpf_ :" << odom_lpf_.transpose());
  }
}

void BasicFrankaController::stopping(const ros::Time& time){
    ROS_INFO("Robot Controller::stopping");
} 

void BasicFrankaController::panda_load_parameter_pub(){              
    kimm_phri_msgs::ObjectParameter objparam_msg;

    objparam_msg.com.resize(3);
    objparam_msg.inertia.resize(9);     

    objparam_msg.mass = m_load_;

    objparam_msg.com[0] = F_x_Cload_[0];
    objparam_msg.com[1] = F_x_Cload_[1];
    objparam_msg.com[2] = F_x_Cload_[2];

    objparam_msg.inertia[0] = I_load_[0];
    objparam_msg.inertia[1] = I_load_[1];
    objparam_msg.inertia[2] = I_load_[2];
    objparam_msg.inertia[3] = I_load_[3];
    objparam_msg.inertia[4] = I_load_[4];
    objparam_msg.inertia[5] = I_load_[5];
    objparam_msg.inertia[6] = I_load_[6];
    objparam_msg.inertia[7] = I_load_[7];
    objparam_msg.inertia[8] = I_load_[8];

    panda_load_parameter_pub_.publish(objparam_msg);              
}

void BasicFrankaController::ctrltypeCallback(const std_msgs::Int16ConstPtr &msg){
    // calculation_mutex_.lock();
    ROS_INFO("[ctrltypeCallback] %d", msg->data);    
    
    if (isgrasp_){
        isgrasp_=false;
        franka_gripper::MoveGoal goal;
        goal.speed = 0.1;
        goal.width = 0.08;
        gripper_ac_.sendGoal(goal);
    }
    else{
        isgrasp_ = true; 
        franka_gripper::GraspGoal goal;
        franka_gripper::GraspEpsilon epsilon;
        epsilon.inner = 0.02;
        epsilon.outer = 0.05;
        goal.speed = 0.1;
        goal.width = 0.02;
        goal.force = 80.0;
        goal.epsilon = epsilon;
        gripper_grasp_ac_.sendGoal(goal);
    }    
    // calculation_mutex_.unlock();
}
void BasicFrankaController::mobtypeCallback(const std_msgs::Int16ConstPtr &msg){    
    ROS_INFO("[mobtypeCallback] %d", msg->data);
    mob_type_ = msg->data;
}
void BasicFrankaController::asyncCalculationProc(){
  calculation_mutex_.lock();
  
  //franka update --------------------------------------------------//
  ctrl_->franka_update(franka_q_, dq_filtered_);

  //franka update with use of pinocchio::aba algorithm--------------//
  // ctrl_->franka_update(franka_q_, dq_filtered_, robot_tau_);
  // ctrl_->franka_update(franka_q_, dq_filtered_, robot_tau_ - robot_g_ - torque_sensor_bias_);

  calculation_mutex_.unlock();
}

Eigen::Matrix<double, 7, 1> BasicFrankaController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

void BasicFrankaController::setFrankaCommand(){  
  robot_command_msg_.MODE = 1;
  robot_command_msg_.header.stamp = ros::Time::now();
  robot_command_msg_.time = time_;

  for (int i=0; i<7; i++)
      robot_command_msg_.torque[i] = franka_torque_(i);   
}

void BasicFrankaController::getEEState(){
    Vector3d pos;
    Quaterniond q;
    ctrl_->ee_state(pos, q);

    ee_state_msg_.translation.x = pos(0);
    ee_state_msg_.translation.y = pos(1);
    ee_state_msg_.translation.z = pos(2);

    ee_state_msg_.rotation.x = q.x();
    ee_state_msg_.rotation.y = q.y();
    ee_state_msg_.rotation.z = q.z();
    ee_state_msg_.rotation.w = q.w();
    ee_state_pub_.publish(ee_state_msg_);
}

void BasicFrankaController::modeChangeReaderProc(){
  while (!quit_all_proc_)
  {
    char key = getchar();
    key = tolower(key);
    calculation_mutex_.lock();

    int msg = 0; 
    switch (key){
      case 'g': //gravity mode
          msg = 0;
          ctrl_->ctrl_update(msg);
          cout << " " << endl;
          cout << "Gravity mode" << endl;
          cout << " " << endl;
          break;
      case 'h': //home
          msg = 1;
          ctrl_->ctrl_update(msg);
          cout << " " << endl;
          cout << "home position" << endl;
          cout << " " << endl;
          break;
      case 'a': //move ee +0.1x
          msg = 2;
          ctrl_->ctrl_update(msg);
          cout << " " << endl;
          cout << "move ee +0.1 x" << endl;
          cout << " " << endl;
          break;    
        case 'q': //move ee -0.1x
          msg = 30;
          ctrl_->ctrl_update(msg);          

          cout << " " << endl;
          cout << "move ee -0.3 x" << endl;
          cout << " " << endl;
          break;       
      case 't': //f_ext test
          msg = 20;
          ctrl_->ctrl_update(msg);
          mob_type_ = 1;

          cout << " " << endl;
          cout << "f_ext test" << endl;
          cout << " " << endl;
          break;       
      case 'y': //f_ext test
          msg = 20;
          ctrl_->ctrl_update(msg);
          mob_type_ = 2;

          cout << " " << endl;
          cout << "f_ext test" << endl;
          cout << " " << endl;
          break;       
      case 'u': //f_ext test
          msg = 20;
          ctrl_->ctrl_update(msg);
          mob_type_ = 0;

          cout << " " << endl;
          cout << "f_ext test" << endl;
          cout << " " << endl;
          break;             
      case 'p': //print current EE state
          msg = 99;
          ctrl_->ctrl_update(msg);
          cout << " " << endl;
          cout << "print current EE state" << endl;
          cout << " " << endl;
          break;      
      case 'z': //grasp
          ROS_INFO_STREAM("Gripper control is locked during object estimation procedure");

          // msg = 899;
          // if (isgrasp_){           
          //     cout << "Release hand" << endl;
          //     isgrasp_ = false;
          //     franka_gripper::MoveGoal goal;
          //     goal.speed = 0.1;
          //     goal.width = 0.08;
          //     gripper_ac_.sendGoal(goal);
          // }
          // else{
          //     cout << "Grasp object" << endl;
          //     isgrasp_ = true; 
          //     franka_gripper::GraspGoal goal;
          //     franka_gripper::GraspEpsilon epsilon;
          //     epsilon.inner = 0.02;
          //     epsilon.outer = 0.05;
          //     goal.speed = 0.1;
          //     goal.width = 0.02;
          //     goal.force = 80.0;
          //     goal.epsilon = epsilon;
          //     gripper_grasp_ac_.sendGoal(goal);
          // }
          break;
      case '\n':
        break;
      case '\r':
        break;
      default:       
        break;
    }
    
    calculation_mutex_.unlock();
  }
}

} // namespace kimm_franka_controllers

PLUGINLIB_EXPORT_CLASS(kimm_franka_controllers::BasicFrankaController, controller_interface::ControllerBase)
