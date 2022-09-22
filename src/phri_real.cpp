
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

  // object estimation 
  object_parameter_pub_ = node_handle.advertise<kimm_phri_msgs::ObjectParameter>("/" + group_name_ + "/real_robot/object_parameter", 5);
  wrench_mesured_pub_ = node_handle.advertise<geometry_msgs::Wrench>("/" + group_name_ + "/real_robot/wrench_measured", 5);
  vel_pub_ = node_handle.advertise<geometry_msgs::Twist>("/" + group_name_ + "/real_robot/object_velocity", 5);
  accel_pub_ = node_handle.advertise<geometry_msgs::Twist>("/" + group_name_ + "/real_robot/object_acceleration", 5);
 
  isgrasp_ = false;  

  // ************ object estimation *************** //               
  isstartestimation = false;
  getObjParam_init();
  // ********************************************** //    
  
  gripper_ac_.waitForServer();
  gripper_grasp_ac_.waitForServer();

  std::vector<std::string> joint_names;
  std::string arm_id;
  ROS_WARN(
      "ForceExampleController: Make sure your robot's endeffector is in contact "
      "with a horizontal surface before starting the controller!");
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("ForceExampleController: Could not read parameter arm_id");
    return false;
  }
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "ForceExampleController: Invalid or no joint_names parameters provided, aborting "
        "controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM("ForceExampleController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "ForceExampleController: Exception getting model handle from interface: " << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM("ForceExampleController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "ForceExampleController: Exception getting state handle from interface: " << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM("ForceExampleController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM("ForceExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }  

  //keyboard event
  mode_change_thread_ = std::thread(&BasicFrankaController::modeChangeReaderProc, this);

  ctrl_ = new RobotController::FrankaWrapper(group_name_, false, node_handle);
  ctrl_->initialize();
  
  
  return true;
}

void BasicFrankaController::starting(const ros::Time& time) {
  dq_filtered_.setZero();

  time_ = 0.;
  dt_ = 0.001;

  mob_type_ = 0;

  robot_command_msg_.torque.resize(7); // 7 (franka) 
  robot_state_msg_.position.resize(9); // 7 (franka) + 2 (gripper)
  robot_state_msg_.velocity.resize(9); // 7 (franka) + 2 (gripper)  

  f_filtered_.setZero();  
}

void BasicFrankaController::update(const ros::Time& time, const ros::Duration& period) {
  
  //update franka variables----------------------------------------------------------------------//
  franka::RobotState robot_state = state_handle_->getRobotState();

  std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  std::array<double, 7> gravity_array = model_handle_->getGravity();
  std::array<double, 49> massmatrix_array = model_handle_->getMass();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();

  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(robot_state.tau_J_d.data());
  robot_tau_ = tau_J_d;

  Eigen::Map<Vector7d> gravity(gravity_array.data());
  robot_g_ = gravity;
  Eigen::Map<Matrix7d> mass_matrix(massmatrix_array.data());
  robot_mass_ = mass_matrix;
  Eigen::Map<Vector7d> non_linear(coriolis_array.data());
  robot_nle_ = non_linear;
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  robot_J_ = jacobian;
  Eigen::Map<Vector7d> franka_q(robot_state.q.data());
  Eigen::Map<Vector7d> franka_dq(robot_state.dq.data());
  franka_q_ = franka_q;
  
  Eigen::Map<Eigen::Matrix<double, 6, 1>> force_franka(robot_state.O_F_ext_hat_K.data());
  f_ = force_franka;

  //filtering ---------------------------------------------------------------------------------//
  double cutoff = 20.0; // Hz //20
  double RC = 1.0 / (cutoff * 2.0 * M_PI);
  double dt = 0.001;
  double alpha = dt / (RC + dt);
  
  dq_filtered_ = alpha * franka_dq + (1 - alpha) * dq_filtered_;
  f_filtered_ = alpha * f_ + (1 - alpha) * f_filtered_;    

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
  ctrl_->compute(time_);  
  ctrl_->franka_output(franka_qacc_); 
  // ctrl_->state(state_);                    //not used now

  //for object estimation
  ctrl_->g_joint7(robot_g_local_);         


  // ctrl_->mass(robot_mass_);                 use franka api mass, not pinocchio mass
  robot_mass_(4, 4) *= 6.0;
  robot_mass_(5, 5) *= 6.0;
  robot_mass_(6, 6) *= 10.0;
  
  franka_torque_ = robot_mass_ * franka_qacc_ + robot_nle_;

  MatrixXd Kd(7, 7);
  Kd.setIdentity();
  Kd = 2.0 * sqrt(5.0) * Kd;
  Kd(5, 5) = 0.2;
  Kd(4, 4) = 0.2;
  Kd(6, 6) = 0.2; // this is practical term
  franka_torque_ -= Kd * dq_filtered_;  
  franka_torque_ << this->saturateTorqueRate(franka_torque_, robot_tau_);

  double thres = 1.0;
  if (ctrl_->ctrltype() != 0){
    if (mob_type_ == 1){
      franka_torque_ -= robot_J_.transpose().col(0) * f_filtered_(0) * thres;
      franka_torque_ -= robot_J_.transpose().col(1) * f_filtered_(1);
      franka_torque_ += robot_J_.transpose().col(2) * f_filtered_(2);
    }
    else if (mob_type_ == 2){
      franka_torque_ += robot_J_.transpose().col(0) * f_filtered_(0);
      franka_torque_ += robot_J_.transpose().col(1) * f_filtered_(1);
      franka_torque_ += robot_J_.transpose().col(2) * f_filtered_(2);
    }
  }

  //send control input to franka
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

  //Object estimation --------------------------------------------------------------//
  // ************ object estimation *************** //               
  vel_accel_pub();        
  FT_measured_pub();        
  getObjParam();                                    // object estimation
  ObjectParameter_pub();                                       // data plot for monitoring
  // ********************************************** //    

  //Debug ------------------------------------------------------------------------//
  // if (print_rate_trigger_())
  // {
  //   ROS_INFO("--------------------------------------------------");
  //   ROS_INFO_STREAM("odom_lpf_ :" << odom_lpf_.transpose());
  // }
}

void BasicFrankaController::stopping(const ros::Time& time){
    ROS_INFO("Robot Controller::stopping");
}

// ************************************************ object estimation start *************************************************** //                       
void BasicFrankaController::vel_accel_pub(){
    geometry_msgs::Twist vel_msg, accel_msg;   

    //************* obtained from pinocchio ***********// 
    //**** velocity (data.v) is identical with mujoco velocity, 
    //**** but acceleration is not reasnoble for both local and global cases.
    // // ctrl_->velocity_global(vel_param);       //offset applied 
    // // ctrl_->acceleration_global(acc_param);   //offset applied 
    // ctrl_->velocity(vel_param);       //offset applied 
    // ctrl_->acceleration(acc_param);   //offset applied 

    //************* obtained from mujoco : LOCAL **************//
    vel_param.linear()[0] = 0.0;
    vel_param.linear()[1] = 0.0;
    vel_param.linear()[2] = 0.0;
    vel_param.angular()[0] = 0.0;
    vel_param.angular()[1] = 0.0;
    vel_param.angular()[2] = 0.0;

    acc_param.linear()[0] = 0.0;
    acc_param.linear()[1] = 0.0;
    acc_param.linear()[2] = 0.0;
    acc_param.angular()[0] = 0.0;
    acc_param.angular()[1] = 0.0;
    acc_param.angular()[2] = 0.0;

    //************************ publish ************************//
    vel_msg.linear.x = vel_param.linear()[0];
    vel_msg.linear.y = vel_param.linear()[1];
    vel_msg.linear.z = vel_param.linear()[2];
    vel_msg.angular.x = vel_param.angular()[0];
    vel_msg.angular.y = vel_param.angular()[1];
    vel_msg.angular.z = vel_param.angular()[2];

    accel_msg.linear.x = acc_param.linear()[0];
    accel_msg.linear.y = acc_param.linear()[1];
    accel_msg.linear.z = acc_param.linear()[2];
    accel_msg.angular.x = acc_param.angular()[0];
    accel_msg.angular.y = acc_param.angular()[1];
    accel_msg.angular.z = acc_param.angular()[2];

    vel_pub_.publish(vel_msg);
    accel_pub_.publish(accel_msg);
}

void BasicFrankaController::FT_measured_pub() {
    FT_measured << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    geometry_msgs::Wrench FT_measured_msg;  
    FT_measured_msg.force.x = saturation(FT_measured[0],50);
    FT_measured_msg.force.y = saturation(FT_measured[1],50);
    FT_measured_msg.force.z = saturation(FT_measured[2],50);
    FT_measured_msg.torque.x = saturation(FT_measured[3],10);
    FT_measured_msg.torque.y = saturation(FT_measured[4],10);
    FT_measured_msg.torque.z = saturation(FT_measured[5],10);
    wrench_mesured_pub_.publish(FT_measured_msg);    
}

void BasicFrankaController::getObjParam(){
    // *********************************************************************** //
    // ************************** object estimation ************************** //
    // *********************************************************************** //
    // mujoco output     : JointStateCallback (motion) : 9 = 7(joint) + 2(gripper)  for pose & velocity & effort, same with rviz jointstate
    // pinocchio input   : state_.q_, v_, dv_ (motion) : 7 = 7(joint)               for q,v,dv
    // controller output : state_.torque_     (torque) : 7 = 7(joint)               for torque, pinocchio doesn't control gripper
    // mujoco input      : robot_command_msg_ (torque) : 9 = 7(joint) + 2(gripper)  robot_command_msg_.torque.resize(9); 
    
    //*--- p cross g = (py*gz-pz*gy)i + (pz*gx-px*gz)j + (px*gy-py*gx)k ---*//

    //*--- FT_measured & vel_param & acc_param is LOCAL frame ---*//

    if (isstartestimation) {

        h = objdyn.h(param, vel_param.toVector(), acc_param.toVector(), robot_g_local_); //Vector3d(0,0,9.81)
        H = objdyn.H(param, vel_param.toVector(), acc_param.toVector(), robot_g_local_); //Vector3d(0,0,9.81)

        ekf->update(FT_measured, dt_, A, H, h);
        param = ekf->state();   

        if (fabs(fabs(robot_g_local_(0)) - 9.81) < 0.02) param[1] = 0.0; //if x axis is aligned with global gravity axis, corresponding param is not meaninful 
        if (fabs(fabs(robot_g_local_(1)) - 9.81) < 0.02) param[2] = 0.0; //if y axis is aligned with global gravity axis, corresponding param is not meaninful 
        if (fabs(fabs(robot_g_local_(2)) - 9.81) < 0.02) param[3] = 0.0; //if z axis is aligned with global gravity axis, corresponding param is not meaninful 
    }
}

void BasicFrankaController::ObjectParameter_pub(){
    kimm_phri_msgs::ObjectParameter objparam_msg;  
    objparam_msg.com.resize(3);
    objparam_msg.inertia.resize(6);    

    Eigen::Vector3d com_global;
    pinocchio::SE3 oMi;
    ctrl_->position(oMi);
    oMi.translation().setZero();
    com_global = oMi.act(Vector3d(param[1], param[2], param[3])); //local to global         
    
    // com_global[2] += (0.1654-0.035); //from l_husky_with_panda_hand.xml, (0.1654:l_panda_rightfinger pos, 0.035: l_panda_rightfinger's cls pos)    

    objparam_msg.mass = saturation(param[0],5.0);
    // objparam_msg.com[0] = saturation(param[1]*sin(M_PI/4)+param[2]*sin(M_PI/4),0.3); // need to check for transformation
    // objparam_msg.com[1] = saturation(param[1]*cos(M_PI/4)-param[2]*cos(M_PI/4),0.3); // need to check for transformation
    // objparam_msg.com[2] = saturation(param[3],0.3);                                  // need to check for transformation
    objparam_msg.com[0] = saturation(com_global[0],0.6); // need to check for transformation
    objparam_msg.com[1] = saturation(com_global[1],0.6); // need to check for transformation
    objparam_msg.com[2] = saturation(com_global[2],0.6); // need to check for transformation

    object_parameter_pub_.publish(objparam_msg);              
}

void BasicFrankaController::getObjParam_init(){
    n_param = 10;
    m_FT = 6;

    A.resize(n_param, n_param); // System dynamics matrix
    H.resize(m_FT,    n_param); // Output matrix
    Q.resize(n_param, n_param); // Process noise covariance
    R.resize(m_FT,    m_FT); // Measurement noise covariance
    P.resize(n_param, n_param); // Estimate error covariance
    h.resize(m_FT,    1); // observation      
    param.resize(n_param);    
    FT_measured.resize(m_FT);        
    
    A.setIdentity();         //knwon, identity
    Q.setIdentity();         //design parameter
    R.setIdentity();         //design parameter    
    P.setIdentity();         //updated parameter
    h.setZero();             //computed parameter
    H.setZero();             //computed parameter    
    param.setZero();   
    FT_measured.setZero();
    vel_param.setZero();
    acc_param.setZero();    

    Q(0,0) *= 0.01;
    Q(1,1) *= 0.0001;
    Q(2,2) *= 0.0001;
    Q(3,3) *= 0.0001;
    R *= 100000;
    
    // Construct the filter
    ekf = new EKF(dt_, A, H, Q, R, P, h);
    
    // Initialize the filter  
    ekf->init(time_, param);
}  

double BasicFrankaController::saturation(double x, double limit) {
    if (x > limit) return limit;
    else if (x < -limit) return -limit;
    else return x;
}
// ************************************************ object estimation end *************************************************** //                       

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
    // calculation_mutex_.lock();
    ROS_INFO("[mobtypeCallback] %d", msg->data);
    mob_type_ = msg->data;
    // if (mob_type_ == 1)
    //   ctrl_->ctrl_update(888);
    
    // calculation_mutex_.unlock();
}
void BasicFrankaController::asyncCalculationProc(){
  calculation_mutex_.lock();
  
  ctrl_->franka_update(franka_q_, dq_filtered_);

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
      case 's': //home and axis align btw base and joint 7
          msg = 3;
          ctrl_->ctrl_update(msg);
          cout << " " << endl;
          cout << "home and axis align btw base and joint 7" << endl;
          cout << " " << endl;
          break;    
      case 'd': //rotate ee
          msg = 4;
          ctrl_->ctrl_update(msg);
          cout << " " << endl;
          cout << "rotate ee" << endl;
          cout << " " << endl;
          break;  
      case 'f': //sine motion ee
          msg = 5;
          ctrl_->ctrl_update(msg);
          cout << " " << endl;
          cout << "sine motion ee" << endl;
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
      case 'o': //object estimation
          if (isstartestimation){
              cout << "end estimation" << endl;
              isstartestimation = false;
              param.setZero();
              ekf->init(time_, param);
          }
          else{
              cout << "start estimation" << endl;
              isstartestimation = true; 
          }
          break;       
      case 'p': //print current EE state
          msg = 99;
          ctrl_->ctrl_update(msg);
          cout << " " << endl;
          cout << "print current EE state" << endl;
          cout << " " << endl;
          break;      
      case 'z': //grasp
          msg = 899;
          if (isgrasp_){
              cout << "Release hand" << endl;
              isgrasp_ = false;
              franka_gripper::MoveGoal goal;
              goal.speed = 0.1;
              goal.width = 0.08;
              gripper_ac_.sendGoal(goal);
          }
          else{
              cout << "Grasp object" << endl;
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
