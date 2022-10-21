
#include <kimm_phri_panda/phri_estimation.h>
#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka/robot_state.h>

namespace kimm_franka_controllers
{

bool ObjectParameterEstimator::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle)
{

  n_node_ = node_handle; //for stopping function
  node_handle.getParam("/robot_group", group_name_);  
  
  ctrl_type_sub_ = node_handle.subscribe("/" + group_name_ + "/real_robot/ctrl_type", 1, &ObjectParameterEstimator::ctrltypeCallback, this);
  mob_subs_ = node_handle.subscribe("/" + group_name_ + "/real_robot/mob_type", 1, &ObjectParameterEstimator::mobtypeCallback, this);
  
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

  // set load
  setload_client = node_handle.serviceClient<franka_msgs::SetLoad>("/franka_control/set_load");  
  flag_ = true;
 
  isgrasp_ = false;  

  // ************ object estimation *************** //               
  isstartestimation = false;
  is_Fext_coordinate_global_ = false; //true : global, false : local
  this->getObjParam_init();

  ROS_INFO_STREAM("force coordinate" << "  " << is_Fext_coordinate_global_);
  // ********************************************** //    
  
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

  //keyboard event
  mode_change_thread_ = std::thread(&ObjectParameterEstimator::modeChangeReaderProc, this);

  ctrl_ = new RobotController::FrankaWrapper(group_name_, false, node_handle);
  ctrl_->initialize(); 

  //ros controller manager carry out starting procedure as start-stop-start sequennce, so prevent stop function
  repeatavoiding_flag_ = false;

  //dynamic_reconfigure    
  ekf_param_node_ = ros::NodeHandle("efk_param_node");
  ekf_param_ = std::make_unique<dynamic_reconfigure::Server<kimm_phri_panda::ekf_paramConfig>>(ekf_param_node_);
  ekf_param_->setCallback(boost::bind(&ObjectParameterEstimator::ekfParamCallback, this, _1, _2));

  return true;
}

void ObjectParameterEstimator::starting(const ros::Time& time) {  
  ROS_INFO("Robot Controller::starting");

  time_ = 0.;
  dt_ = 0.001;

  mob_type_ = 0;

  robot_command_msg_.torque.resize(7); // 7 (franka) 
  robot_state_msg_.position.resize(9); // 7 (franka) + 2 (gripper)
  robot_state_msg_.velocity.resize(9); // 7 (franka) + 2 (gripper)  

  dq_filtered_.setZero();
  f_filtered_.setZero();    
  f_local_filtered_.setZero();    

  //franka state_handle -------------------------------//
  franka::RobotState robot_state = state_handle_->getRobotState(); 

  double m_load(robot_state.m_load);
  m_load_ = m_load; //external load's mass.   
  
  ROS_INFO_STREAM("m_load_ :" << m_load_);
}

void ObjectParameterEstimator::update(const ros::Time& time, const ros::Duration& period) {  

  // ROS_INFO("I am here1");
  // modeChangeReaderProc();
  // ROS_INFO("I am here4");
  
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

  Eigen::Map<Eigen::Matrix<double, 6, 1>> force_franka_local(robot_state.K_F_ext_hat_K.data());
  f_local_ = force_franka_local; //Estimated external wrench (force, torque) acting on stiffness frame, expressed relative to the stiffness frame. 

  double m_load(robot_state.m_load);
  m_load_ = m_load; //external load's mass.   

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
  f_local_filtered_ = lowpassFilter( dt_,  f_local_,   f_local_filtered_,  20.0); //in Hz, Vector6d  
  
  // thread for franka state update to HQP -----------------------------------------------------//
  if (calculation_mutex_.try_lock())
  {
    calculation_mutex_.unlock();
    if (async_calculation_thread_.joinable())
      async_calculation_thread_.join();

    //asyncCalculationProc -->  ctrl_->franka_update(franka_q_, dq_filtered_);
    async_calculation_thread_ = std::thread(&ObjectParameterEstimator::asyncCalculationProc, this);        
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
  ctrl_->g_joint7(robot_g_local_);    
  ctrl_->JLocal_offset(robot_J_local_);     //offset applied ,local joint7
  ctrl_->dJLocal_offset(robot_dJ_local_);   //offset applied ,local joint7

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

  //Object estimation --------------------------------------------------------------//
  // ************ object estimation *************** //              
  this->vel_accel_pub();        
  this->FT_measured_pub();        
  this->getObjParam();                                    // object estimation
  this->ObjectParameter_pub();                            // data plot for monitoring
  // ********************************************** //    

  //Debug ------------------------------------------------------------------------//
  if (print_rate_trigger_())
  {
    // ROS_INFO("--------------------------------------------------");
    // ROS_INFO_STREAM("robot_mass_ :" << robot_mass_);
    // ROS_INFO_STREAM("m_load_ :" << m_load_);
    // ROS_INFO_STREAM("odom_lpf_ :" << odom_lpf_.transpose());
    // ROS_INFO_STREAM("robot_g_local" << robot_g_local_(0) << robot_g_local_(1) << robot_g_local_(2));

  }
}

void ObjectParameterEstimator::stopping(const ros::Time& time){
    ROS_INFO("Robot Controller::stopping");

    if (repeatavoiding_flag_) //do it when true
    {
      // double obj_mass;
      // std::vector<double> obj_com, obj_inertia;

      // obj_mass = 0.05;

      // obj_com.resize(3);
      // obj_com[0] = 0.0;
      // obj_com[1] = 0.0;
      // obj_com[2] = 0.0;

      // obj_inertia.resize(9);
      // obj_inertia[0] = 0.0;
      // obj_inertia[1] = 0.0;
      // obj_inertia[2] = 0.0;
      // obj_inertia[3] = 0.0;
      // obj_inertia[4] = 0.0;
      // obj_inertia[5] = 0.0;
      // obj_inertia[6] = 0.0;
      // obj_inertia[7] = 0.0;
      // obj_inertia[8] = 0.0;

      // n_node_.setParam("/object_parameter/mass",obj_mass);
      // n_node_.setParam("/object_parameter/center_of_mass",obj_com);
      // n_node_.setParam("/object_parameter/inertia",obj_inertia);
    }
    else //first stop step
    {
      repeatavoiding_flag_ =  true;
    }            
} 

// ************************************************ object estimation start *************************************************** //                       
void ObjectParameterEstimator::setObjParam() {  
  // param = [m, cx, cy, cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]

  double obj_mass;
  std::vector<double> obj_com, obj_inertia;
  
  obj_mass = param[0];

  obj_com.resize(3);
  obj_com[0] = param[1];
  obj_com[1] = param[2];
  obj_com[2] = param[3];

  obj_inertia.resize(9);
  // obj_inertia[0] = param[4];
  // obj_inertia[1] = param[5];
  // obj_inertia[2] = param[6];  
  // obj_inertia[3] = param[5];
  // obj_inertia[4] = param[7];
  // obj_inertia[5] = param[8];  
  // obj_inertia[6] = param[6];
  // obj_inertia[7] = param[8];
  // obj_inertia[8] = param[9];

  obj_inertia[0] = 0.0;
  obj_inertia[1] = 0.0;
  obj_inertia[2] = 0.0;  
  obj_inertia[3] = 0.0;
  obj_inertia[4] = 0.0;
  obj_inertia[5] = 0.0;  
  obj_inertia[6] = 0.0;
  obj_inertia[7] = 0.0;
  obj_inertia[8] = 0.0;

  n_node_.setParam("/object_parameter/mass",obj_mass);
  n_node_.setParam("/object_parameter/center_of_mass",obj_com);
  n_node_.setParam("/object_parameter/inertia",obj_inertia);
}

void ObjectParameterEstimator::ekfParamCallback(kimm_phri_panda::ekf_paramConfig& config, uint32_t level) {
  Q(0,0) = config.Q0;  
  Q(1,1) = config.Q1;  
  Q(2,2) = config.Q2;  
  Q(3,3) = config.Q3;  
  Q(4,4) = config.Q4;  
  Q(5,5) = config.Q5;  
  Q(6,6) = config.Q6;  
  Q(7,7) = config.Q7;  
  Q(8,8) = config.Q8;  
  Q(9,9) = config.Q9;  

  R(0,0) = config.R0;  
  R(1,1) = config.R1;  
  R(2,2) = config.R2;  
  R(3,3) = config.R3;  
  R(4,4) = config.R4;  
  R(5,5) = config.R5;  
  
  ROS_INFO("--------------------------------------------------");
  ROS_INFO_STREAM("Q" << "  " << Q(0,0) << "  " << Q(1,1) << "  " << Q(2,2) << "  " << Q(3,3) );
  ROS_INFO_STREAM("Q" << "  " << Q(4,4) << "  " << Q(5,5) << "  " << Q(6,6) << "  " << Q(7,7) << "  " << Q(8,8) << "  " << Q(9,9));
  ROS_INFO_STREAM("R" << "  " << R(0,0) << "  " << R(1,1) << "  " << R(2,2) << "  " << R(3,3) << "  " << R(4,4) << "  " << R(5,5) );
}

void ObjectParameterEstimator::vel_accel_pub(){
    //************* obtained from pinocchio ***********// 
    //**** velocity (data.v) is identical with franka velocity dq_filtered_, 
    //**** but acceleration (data.a) is not reasnoble for both local and global cases.
    // ctrl_->velocity(vel_param);                //offset is applied, LOCAL
    // ctrl_->acceleration(acc_param);            //offset is applied, LOCAL
    
    // ctrl_->velocity_global(vel_param);         //offset is applied, GLOBAL
    // ctrl_->acceleration_global(acc_param);     //offset is applied, GLOBAL        
    
    // ctrl_->velocity_origin(vel_param);         //offset is not applied, GLOBAL  --> if offset is zero, velocity_global = velocity_orign
    // ctrl_->acceleration(acc_param);            //offset not is applied, GLOBAL  --> if offset is zero, acceleration_global = acceleration_orign    

    //************* obtained from franka and pinocchio : LOCAL **************//            
    if (isstartestimation) { //start only when the robot is in proper state

      if(tau_bias_init_flag) {   //due to the bias of torque sensor in franka, initial bias should be corrected when estimation is started (See force_example_controller.cpp in franka_ros)
        torque_sensor_bias_ = robot_tau_ - robot_g_;  
        tau_bias_init_flag = false;
      }      

      //ddq from this method is not a acceleration behavior, but torque behavior (bias is presented in case of the no motion) ------------------//
      // franka_ddq_for_param_ = robot_mass_.inverse() * (robot_tau_ - robot_g_ - torque_sensor_bias_) - robot_nle_; 
      
      //ddq from derivative of dq is very noisy, but for now, there is no other way to obatin ddq (0930) ---------------------------------------//
      franka_ddq_for_param_ = (franka_dq_ - franka_dq_prev_)/dt_;
      franka_dq_prev_ = franka_dq_;

      //obtain cartesian velocity and acceleration at joint7 LOCAL frame -----------------------------------------------------------------------//
      franka_v_.setZero();
      franka_a_.setZero();
      for (int i=0; i<7; i++){         
          franka_v_ += robot_J_local_.col(i) * dq_filtered_[i];                                                     
          franka_a_ += robot_dJ_local_.col(i) * franka_dq_[i] + robot_J_local_.col(i) * franka_ddq_for_param_[i];   
      }              
    }
    else { //no estimation state
      torque_sensor_bias_.setZero();
      franka_ddq_for_param_.setZero();
      franka_a_.setZero();

      franka_v_.setZero();
      for (int i=0; i<7; i++){ //dq is measured signal so, show regradless of estimation on/off
          franka_v_ += robot_J_local_.col(i) * dq_filtered_[i];                                                     
      }              
    }

    // Filtering
    franka_a_filtered_ = lowpassFilter( dt_, franka_a_, franka_a_filtered_, 5.0); //in Hz, Vector6d

    //convert to pinocchio::Motion -----------------------------------------------------------------------------------------------------------//
    vel_param.linear()[0] = franka_v_(0);
    vel_param.linear()[1] = franka_v_(1);
    vel_param.linear()[2] = franka_v_(2);            
    vel_param.angular()[0] = franka_v_(3);
    vel_param.angular()[1] = franka_v_(4);
    vel_param.angular()[2] = franka_v_(5);

    acc_param.linear()[0] = franka_a_filtered_(0);
    acc_param.linear()[1] = franka_a_filtered_(1);
    acc_param.linear()[2] = franka_a_filtered_(2);
    acc_param.angular()[0] = franka_a_filtered_(3);
    acc_param.angular()[1] = franka_a_filtered_(4);
    acc_param.angular()[2] = franka_a_filtered_(5);

    //publish --------------------------------------------------------------------------------------------------------------------------------//
    geometry_msgs::Twist vel_msg, accel_msg;   

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

void ObjectParameterEstimator::FT_measured_pub() {    
    for (int i=0; i<6; i++){  
      //obtained from franka API--------------------------------//
      // FT_measured[i] = f_filtered_(i);        //GLOBAL 
      // FT_measured[i] = -f_local_filtered_(i); //LOCAL w.r.t. hand frame, not joint7 frame
    }

    //obtained from pinocchio for the joint7 local frame--------//
    pinocchio::SE3 oMi;
    pinocchio::Force f_pin_global, f_pin_local;
    Vector6d f_used;

    if (is_Fext_coordinate_global_) //global
    {
      f_used = f_filtered_;      
      
      if (isstartestimation) { //start only when the robot is in proper state
        if(F_ext_bias_init_flag) {   //F_ext bias to be corrected 
          F_ext_bias_ = f_used;  
          F_ext_bias_init_flag = false;
        } 
      }
      else {
        F_ext_bias_.setZero();
      }     

      //GLOBAL ---------------------------------------//
      f_pin_global.linear()[0] = f_used(0) - F_ext_bias_(0);
      f_pin_global.linear()[1] = f_used(1) - F_ext_bias_(1);
      f_pin_global.linear()[2] = f_used(2) - F_ext_bias_(2);
      f_pin_global.angular()[0] = f_used(3) - F_ext_bias_(3);
      f_pin_global.angular()[1] = f_used(4) - F_ext_bias_(4);
      f_pin_global.angular()[2] = f_used(5) - F_ext_bias_(5);

      //LOCAL ---------------------------------------//
      ctrl_->position(oMi);
      // oMi.translation().setZero();    

      f_pin_local = -oMi.actInv(f_pin_global); //global to local, f is external forces, so (-) is needed
      // f_pin_local = oMi.actInv(f_pin_global); //global to local      
    }
    else //local
    {      
      f_used = f_local_filtered_;

      if (isstartestimation) { //start only when the robot is in proper state
        if(F_ext_bias_init_flag) {   //F_ext bias to be corrected 
          F_ext_bias_ = f_used;  
          F_ext_bias_init_flag = false;
        } 
      }
      else {
        F_ext_bias_.setZero();
      }     

      //LOCAL stiffness frame ------------------------//
      f_pin_global.linear()[0] = f_used(0) - F_ext_bias_(0);
      f_pin_global.linear()[1] = f_used(1) - F_ext_bias_(1);
      f_pin_global.linear()[2] = f_used(2) - F_ext_bias_(2);
      f_pin_global.angular()[0] = f_used(3) - F_ext_bias_(3);
      f_pin_global.angular()[1] = f_used(4) - F_ext_bias_(4);
      f_pin_global.angular()[2] = f_used(5) - F_ext_bias_(5);

      //stiffness frame to joint7  ------------------//      
      // transfrom SE to joint7 with only rotation, not translation
      oMi.translation().setZero();    
      oMi.rotation().setIdentity();
      
      // rotation from joint 7 to SE
      oMi.rotation()(0,0) =  0.707;
      oMi.rotation()(0,1) =  0.707;
      oMi.rotation()(1,0) = -0.707;
      oMi.rotation()(1,1) =  0.707;

      f_pin_local = -oMi.act(f_pin_global); //global to local, f is external forces, so (-) is needed      
    }

    FT_measured[0] = f_pin_local.linear()[0];
    FT_measured[1] = f_pin_local.linear()[1];
    FT_measured[2] = f_pin_local.linear()[2];
    FT_measured[3] = f_pin_local.angular()[0];
    FT_measured[4] = f_pin_local.angular()[1];
    FT_measured[5] = f_pin_local.angular()[2];

    //publish ---------------------------------------------------//
    geometry_msgs::Wrench FT_measured_msg;  

    FT_measured_msg.force.x = saturation(FT_measured[0],50);
    FT_measured_msg.force.y = saturation(FT_measured[1],50);
    FT_measured_msg.force.z = saturation(FT_measured[2],50);
    FT_measured_msg.torque.x = saturation(FT_measured[3],10);
    FT_measured_msg.torque.y = saturation(FT_measured[4],10);
    FT_measured_msg.torque.z = saturation(FT_measured[5],10);

    wrench_mesured_pub_.publish(FT_measured_msg);    
}

void ObjectParameterEstimator::getObjParam(){
    // *********************************************************************** //
    // ************************** object estimation ************************** //
    // *********************************************************************** //
    // franka output     : robot_state        (motion) : 7 = 7(joint)               for pose & velocity, same with rviz jointstate
    // pinocchio input   : state_.q_, v_      (motion) : 7 = 7(joint)               for q,v
    // controller output : state_.torque_     (torque) : 7 = 7(joint)               for torque, pinocchio doesn't control gripper
    // franka input      : setCommand         (torque) : 7 = 7(joint)               gripper is controlle by other action client
    
    //*--- p cross g = (py*gz-pz*gy)i + (pz*gx-px*gz)j + (px*gy-py*gx)k ---*//

    //*--- FT_measured & vel_param & acc_param is joint7 LOCAL frame ---*//

    if (isstartestimation) {

        h = objdyn.h(param, vel_param.toVector(), acc_param.toVector(), robot_g_local_); //Vector3d(0,0,9.81)
        H = objdyn.H(param, vel_param.toVector(), acc_param.toVector(), robot_g_local_); //Vector3d(0,0,9.81)

        // ekf->update(FT_measured, dt_, A, H, h);
        ekf->update(FT_measured, dt_, A, H, h, Q, R); //Q & R update from dynamic reconfigure
        param = ekf->state();   

        if (fabs(fabs(robot_g_local_(0)) - 9.81) < 0.02) param[1] = 0.0; //if x axis is aligned with global gravity axis, corresponding param is not meaninful 
        if (fabs(fabs(robot_g_local_(1)) - 9.81) < 0.02) param[2] = 0.0; //if y axis is aligned with global gravity axis, corresponding param is not meaninful 
        if (fabs(fabs(robot_g_local_(2)) - 9.81) < 0.02) param[3] = 0.0; //if z axis is aligned with global gravity axis, corresponding param is not meaninful 
    }
}

void ObjectParameterEstimator::ObjectParameter_pub(){         
    //LOCAL to GLOBAL -------------------------------------------//
    Eigen::Vector3d com_global;
    pinocchio::SE3 oMi;
    ctrl_->position(oMi);
    oMi.translation().setZero();
    com_global = oMi.act(Vector3d(param[1], param[2], param[3])); //local to global   

    
    // com_global[2] += (0.1654-0.035); //from l_husky_with_panda_hand.xml, (0.1654:l_panda_rightfinger pos, 0.035: l_panda_rightfinger's cls pos)    

    //publish ---------------------------------------------------//
    kimm_phri_msgs::ObjectParameter objparam_msg;

    objparam_msg.com.resize(3);
    objparam_msg.inertia.resize(6);     

    objparam_msg.mass = saturation(param[0],5.0);        
    
    objparam_msg.com[0] = saturation(com_global[0],0.8); // need to check for transformation
    objparam_msg.com[1] = saturation(com_global[1],0.8); // need to check for transformation
    objparam_msg.com[2] = saturation(com_global[2],0.8); // need to check for transformation
    
    // objparam_msg.com[0] = saturation(param[1],1.6); // need to check for transformation
    // objparam_msg.com[1] = saturation(param[2],0.6); // need to check for transformation
    // objparam_msg.com[2] = saturation(param[3],0.6); // need to check for transformation

    object_parameter_pub_.publish(objparam_msg);              
}

void ObjectParameterEstimator::getObjParam_init(){
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
    franka_dq_prev_.setZero();

    // Q(0,0) *= 0.01;
    // Q(1,1) *= 0.0001;
    // Q(2,2) *= 0.0001;
    // Q(3,3) *= 0.0001;
    // R *= 100000;

    // Q(0,0) *= 0.01;
    // Q(1,1) *= 0.0001;
    // Q(2,2) *= 0.0001;
    // Q(3,3) *= 0.0001;
    // R *= 1000;

    Q(0,0) = 0.01; // 10/21 
    Q(1,1) = 0.01;  
    Q(2,2) = 0.01;  
    Q(3,3) = 0.01;  
    Q(4,4) = 0.0;  
    Q(5,5) = 0.0;  
    Q(6,6) = 0.0;  
    Q(7,7) = 0.0;  
    Q(8,8) = 0.0;  
    Q(9,9) = 0.0;  

    R(0,0) = 0.5;  
    R(1,1) = 0.2;  
    R(2,2) = 10.0;  
    R(3,3) = 10.0;  
    R(4,4) = 10.0;  
    R(5,5) = 10.0;  
  
    ROS_INFO("------------------------------ initial ekf parameter ------------------------------");
    ROS_INFO_STREAM("Q" << "  " << Q(0,0) << "  " << Q(1,1) << "  " << Q(2,2) << "  " << Q(3,3) );
    ROS_INFO_STREAM("Q" << "  " << Q(4,4) << "  " << Q(5,5) << "  " << Q(6,6) << "  " << Q(7,7) << "  " << Q(8,8) << "  " << Q(9,9));
    ROS_INFO_STREAM("R" << "  " << R(0,0) << "  " << R(1,1) << "  " << R(2,2) << "  " << R(3,3) << "  " << R(4,4) << "  " << R(5,5) );
    
    // Construct the filter
    ekf = new EKF(dt_, A, H, Q, R, P, h);
    
    // Initialize the filter  
    ekf->init(time_, param);
}  

double ObjectParameterEstimator::saturation(double x, double limit) {
    if (x > limit) return limit;
    else if (x < -limit) return -limit;
    else return x;
}
// ************************************************ object estimation end *************************************************** //                       

void ObjectParameterEstimator::ctrltypeCallback(const std_msgs::Int16ConstPtr &msg){
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
void ObjectParameterEstimator::mobtypeCallback(const std_msgs::Int16ConstPtr &msg){    
    ROS_INFO("[mobtypeCallback] %d", msg->data);
    mob_type_ = msg->data;
}
void ObjectParameterEstimator::asyncCalculationProc(){
  calculation_mutex_.lock();
  
  //franka update --------------------------------------------------//
  ctrl_->franka_update(franka_q_, dq_filtered_);

  //franka update with use of pinocchio::aba algorithm--------------//
  // ctrl_->franka_update(franka_q_, dq_filtered_, robot_tau_);
  // ctrl_->franka_update(franka_q_, dq_filtered_, robot_tau_ - robot_g_ - torque_sensor_bias_);

  calculation_mutex_.unlock();
}

Eigen::Matrix<double, 7, 1> ObjectParameterEstimator::saturateTorqueRate(
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

void ObjectParameterEstimator::setFrankaCommand(){  
  robot_command_msg_.MODE = 1;
  robot_command_msg_.header.stamp = ros::Time::now();
  robot_command_msg_.time = time_;

  for (int i=0; i<7; i++)
      robot_command_msg_.torque[i] = franka_torque_(i);   
}

void ObjectParameterEstimator::getEEState(){
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

void ObjectParameterEstimator::modeChangeReaderProc(){
  // ROS_INFO("I am here2");

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
              tau_bias_init_flag = true;
              F_ext_bias_init_flag = true;
          }
          break;       
      case 'i': //set object parameter
          this->setObjParam();

          cout << " " << endl;
          cout << "set object parameter" << endl;
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
  // ROS_INFO("I am here3");
}

} // namespace kimm_franka_controllers

PLUGINLIB_EXPORT_CLASS(kimm_franka_controllers::ObjectParameterEstimator, controller_interface::ControllerBase)
