#include "kimm_phri_panda/phri_simul.h"

using namespace std;
using namespace pinocchio;
using namespace Eigen;
using namespace RobotController;

int main(int argc, char **argv)
{   
    //Ros setting
    ros::init(argc, argv, "kimm_phri_panda");
    ros::NodeHandle n_node;
    
    dt = 0.001;
    time_ = 0.0;
    ros::Rate loop_rate(1.0/dt);

    /////////////// Robot Wrapper ///////////////
    n_node.getParam("/robot_group", group_name);    
    // ctrl_ = new RobotController::FrankaWrapper(group_name, true, n_node);
    ctrl_ = new RobotController::FrankaWrapper(group_name, true, n_node, 0);
    ctrl_->initialize();
    
    /////////////// mujoco sub : from mujoco to here ///////////////    
    ros::Subscriber jointState = n_node.subscribe("mujoco_ros/mujoco_ros_interface/joint_states", 5, &JointStateCallback, ros::TransportHints().tcpNoDelay(true));    
    ros::Subscriber mujoco_command_sub = n_node.subscribe("mujoco_ros/mujoco_ros_interface/sim_command_sim2con", 5, &simCommandCallback, ros::TransportHints().tcpNoDelay(true));
    ros::Subscriber mujoco_time_sub = n_node.subscribe("mujoco_ros/mujoco_ros_interface/sim_time", 1, &simTimeCallback, ros::TransportHints().tcpNoDelay(true));
    ros::Subscriber ctrl_type_sub = n_node.subscribe("mujoco_ros/mujoco_ros_interface/ctrl_type", 1, &ctrltypeCallback, ros::TransportHints().tcpNoDelay(true));

    /////////////// mujoco pub : from here to mujoco ///////////////    
    mujoco_command_pub_ = n_node.advertise<std_msgs::String>("mujoco_ros/mujoco_ros_interface/sim_command_con2sim", 5);
    robot_command_pub_ = n_node.advertise<mujoco_ros_msgs::JointSet>("mujoco_ros/mujoco_ros_interface/joint_set", 5);
    mujoco_run_pub_ = n_node.advertise<std_msgs::Bool>("mujoco_ros/mujoco_ros_interface/sim_run", 5);

    joint_states_pub_ = n_node.advertise<sensor_msgs::JointState>("joint_states", 5);    
    object_parameter_pub_ = n_node.advertise<kimm_phri_msgs::ObjectParameter>("object_parameter", 5);
    wrench_mesured_pub_ = n_node.advertise<geometry_msgs::Wrench>("wrench_measured", 5);
    vel_pub_ = n_node.advertise<geometry_msgs::Twist>("object_velocity", 5);
    accel_pub_ = n_node.advertise<geometry_msgs::Twist>("object_acceleration", 5);

    /////////////// robot - ctrl(phri_hqp), robot(robot_wrapper) ///////////////        
    ee_state_pub_ = n_node.advertise<geometry_msgs::Transform>("mujoco_ros/mujoco_ros_interface/ee_state", 5);
    
    // msg 
    robot_command_msg_.torque.resize(9);         // robot (7) + gripper(2) --> from here to mujoco    
    ee_state_msg_ = geometry_msgs::Transform();  // obtained from "robot_->position"

    sim_run_msg_.data = true;
    isgrasp_ = false;
    isstartestimation = false;

    // InitMob();

    // ************ object estimation *************** //               
    getObjParam_init();
    // ********************************************** //    

    while (ros::ok()){        
        //mujoco sim run 
        mujoco_run_pub_.publish(sim_run_msg_);
       
        //keyboard
        keyboard_event();

        // ctrl computation
        ctrl_->compute(time_); //make control input for 1kHz, joint state will be updated 1kHz from the mujoco
        
        // get output
        ctrl_->mass(robot_mass_);
        ctrl_->nle(robot_nle_);
        ctrl_->g(robot_g_);  // dim model.nv, [Nm]
        ctrl_->g_joint7(robot_g_local_);  //g [m/s^2] w.r.t joint7 axis
        ctrl_->state(state_);           

        ctrl_->JWorld(robot_J_world_);            //world
        ctrl_->JLocal_offset(robot_J_local_);     //offset applied ,local
        ctrl_->dJLocal_offset(robot_dJ_local_);   //offset applied ,local

        // get control input from hqp controller
        ctrl_->franka_output(franka_qacc_); //get control input
        franka_torque_ = robot_mass_ * franka_qacc_ + robot_nle_; 
        
        // get Mob
        // if (ctrl_->ctrltype() != 0)
        //     UpdateMob();
        // else
        //     InitMob();

        // set control input to mujoco
        setGripperCommand();                              //set gripper torque by trigger value
        setRobotCommand();                                //set franka and husky command 
        robot_command_pub_.publish(robot_command_msg_);   //pub total command
       
        // get state
        getEEState();                                     //obtained from "robot_->position", and publish for monitoring              

        // ************ object estimation *************** //               
        vel_accel_pub();        
        FT_measured_pub();        
        getObjParam();                                    // object estimation
        ObjectParameter_pub();                                       // data plot for monitoring
        // ********************************************** //    
        
        ros::spinOnce();
        loop_rate.sleep();        
    }//while

    return 0;
}

// ************************************************ object estimation start *************************************************** //                       
void vel_accel_pub(){
    geometry_msgs::Twist vel_msg, accel_msg;   

    //************* obtained from pinocchio ***********// 
    //**** velocity (data.v) is identical with mujoco velocity, 
    //**** but acceleration (data.a) is not reasnoble for both local and global cases.
    // ctrl_->velocity(vel_param);                //offset is applied, LOCAL
    // ctrl_->acceleration(acc_param);            //offset is applied, LOCAL
    
    // ctrl_->velocity_global(vel_param);         //offset is applied, GLOBAL
    // ctrl_->acceleration_global(acc_param);     //offset is applied, GLOBAL        
    
    // ctrl_->velocity_origin(vel_param);         //offset is not applied, GLOBAL  --> if offset is zero, velocity_global = velocity_orign
    // ctrl_->acceleration(acc_param);            //offset not is applied, GLOBAL  --> if offset is zero, acceleration_global = acceleration_orign    

    //************* obtained from mujoco : LOCAL **************//
    vel_param.linear()[0] = v_mujoco[0];
    vel_param.linear()[1] = v_mujoco[1];
    vel_param.linear()[2] = v_mujoco[2];
    vel_param.angular()[0] = v_mujoco[3];
    vel_param.angular()[1] = v_mujoco[4];
    vel_param.angular()[2] = v_mujoco[5];

    // acc_param.linear()[0] = a_mujoco[0];
    // acc_param.linear()[1] = a_mujoco[1];
    // acc_param.linear()[2] = a_mujoco[2];
    // acc_param.angular()[0] = a_mujoco[3];
    // acc_param.angular()[1] = a_mujoco[4];
    // acc_param.angular()[2] = a_mujoco[5];

    acc_param.linear()[0] = a_mujoco_filtered[0];
    acc_param.linear()[1] = a_mujoco_filtered[1];
    acc_param.linear()[2] = a_mujoco_filtered[2];
    acc_param.angular()[0] = a_mujoco_filtered[3];
    acc_param.angular()[1] = a_mujoco_filtered[4];
    acc_param.angular()[2] = a_mujoco_filtered[5];

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

void FT_measured_pub() {
    //actually, franka_torque_ is not a measured but command torque, because measurment is not available
    tau_estimated = robot_mass_ * ddq_mujoco + robot_nle_;        
    // tau_ext = franka_torque_ - tau_estimated;      // coincide with g(0,0,-9.81)  
    tau_ext = -franka_torque_ + tau_estimated;        // coincide with g(0,0,9.81)
    FT_measured = robot_J_local_.transpose().completeOrthogonalDecomposition().pseudoInverse() * tau_ext;  //robot_J_local is local jacobian      

    geometry_msgs::Wrench FT_measured_msg;  
    FT_measured_msg.force.x = saturation(FT_measured[0],50);
    FT_measured_msg.force.y = saturation(FT_measured[1],50);
    FT_measured_msg.force.z = saturation(FT_measured[2],50);
    FT_measured_msg.torque.x = saturation(FT_measured[3],10);
    FT_measured_msg.torque.y = saturation(FT_measured[4],10);
    FT_measured_msg.torque.z = saturation(FT_measured[5],10);
    wrench_mesured_pub_.publish(FT_measured_msg);    
}

void getObjParam(){
    // *********************************************************************** //
    // ************************** object estimation ************************** //
    // *********************************************************************** //
    // mujoco output     : JointStateCallback (motion) : 9 = 7(joint) + 2(gripper)  for pose & velocity & effort, same with rviz jointstate
    // pinocchio input   : state_.q_, v_      (motion) : 7 = 7(joint)               for q,v
    // controller output : state_.torque_     (torque) : 7 = 7(joint)               for torque, pinocchio doesn't control gripper
    // mujoco input      : robot_command_msg_ (torque) : 9 = 7(joint) + 2(gripper)  robot_command_msg_.torque.resize(9); 
    
    //*--- p cross g = (py*gz-pz*gy)i + (pz*gx-px*gz)j + (px*gy-py*gx)k ---*//

    //*--- FT_measured & vel_param & acc_param is LOCAL frame ---*//

    if (isstartestimation) {

        h = objdyn.h(param, vel_param.toVector(), acc_param.toVector(), robot_g_local_); //Vector3d(0,0,9.81)
        H = objdyn.H(param, vel_param.toVector(), acc_param.toVector(), robot_g_local_); //Vector3d(0,0,9.81)

        ekf->update(FT_measured, dt, A, H, h);
        param = ekf->state();   

        if (fabs(fabs(robot_g_local_(0)) - 9.81) < 0.02) param[1] = 0.0; //if x axis is aligned with global gravity axis, corresponding param is not meaninful 
        if (fabs(fabs(robot_g_local_(1)) - 9.81) < 0.02) param[2] = 0.0; //if y axis is aligned with global gravity axis, corresponding param is not meaninful 
        if (fabs(fabs(robot_g_local_(2)) - 9.81) < 0.02) param[3] = 0.0; //if z axis is aligned with global gravity axis, corresponding param is not meaninful 
    }
}

void ObjectParameter_pub(){
    kimm_phri_msgs::ObjectParameter objparam_msg;  
    objparam_msg.com.resize(3);
    objparam_msg.inertia.resize(6);    

    Eigen::Vector3d com_global;
    SE3 oMi;
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

void getObjParam_init(){
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
    ddq_mujoco.resize(7);
    tau_estimated.resize(7);
    tau_ext.resize(7);
    v_mujoco.resize(6);
    a_mujoco.resize(6);
    a_mujoco_filtered.resize(6);    

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
    ddq_mujoco.setZero();
    tau_estimated.setZero();
    tau_ext.setZero();
    v_mujoco.setZero();
    a_mujoco.setZero();
    a_mujoco_filtered.setZero();

    Q(0,0) *= 0.01;
    Q(1,1) *= 0.0001;
    Q(2,2) *= 0.0001;
    Q(3,3) *= 0.0001;
    R *= 100000;
    
    // Construct the filter
    ekf = new EKF(dt, A, H, Q, R, P, h);
    
    // Initialize the filter  
    ekf->init(time_, param);
}  

double saturation(double x, double limit) {
    if (x > limit) return limit;
    else if (x < -limit) return -limit;
    else return x;
}
// ************************************************ object estimation end *************************************************** //                       


void simCommandCallback(const std_msgs::StringConstPtr &msg){
    std::string buf;
    buf = msg->data;

    if (buf == "RESET")
    {
        std_msgs::String rst_msg_;
        rst_msg_.data = "RESET";
        mujoco_command_pub_.publish(rst_msg_);
    }

    if (buf == "INIT")
    {
        std_msgs::String rst_msg_;
        rst_msg_.data = "INIT";
        mujoco_command_pub_.publish(rst_msg_);
        mujoco_time_ = 0.0;
    }
}

void simTimeCallback(const std_msgs::Float32ConstPtr &msg){
    mujoco_time_ = msg->data;
    time_ = mujoco_time_;
}

void JointStateCallback(const sensor_msgs::JointState::ConstPtr& msg){ 
    // from mujoco
    // msg.position : 7(joint) + 2(gripper)
    // msg.velocity : 7(joint) + 2(gripper)
    sensor_msgs::JointState msg_tmp;
    msg_tmp = *msg;    

    //update state to pinocchio
    // state_.q_      //7 franka (7)
    // state_.v_      //7
    // state_.dv_     //7
    // state_.torque_ //7  

    ctrl_->franka_update(msg_tmp);        
    joint_states_publish(msg_tmp);        

    v_mujoco.setZero();
    a_mujoco.setZero();    
    for (int i=0; i<7; i++){ 
        ddq_mujoco[i] = msg_tmp.effort[i];
        v_mujoco += robot_J_local_.col(i) * msg_tmp.velocity[i];                                                 //jacobian is LOCAL
        a_mujoco += robot_dJ_local_.col(i) * msg_tmp.velocity[i] + robot_J_local_.col(i) * msg_tmp.effort[i];    //jacobian is LOCAL        
    }        

    // Filtering
    double cutoff = 20.0; // Hz //20
    double RC = 1.0 / (cutoff * 2.0 * M_PI);    
    double alpha = dt / (RC + dt);

    a_mujoco_filtered = alpha * a_mujoco + (1 - alpha) * a_mujoco_filtered;        
}

void joint_states_publish(const sensor_msgs::JointState& msg){
    // mujoco callback msg
    // msg.position : 7(joint) + 2(gripper)
    // msg.velocity : 7(joint) + 2(gripper) 

    sensor_msgs::JointState joint_states;
    joint_states.header.stamp = ros::Time::now();    

    //revolute joint name in rviz urdf (panda_arm_hand_l_rviz.urdf)
    joint_states.name = {"panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7","panda_finger_joint1","panda_finger_joint2"};    

    joint_states.position.resize(9); //panda(7) + finger(2)
    joint_states.velocity.resize(9); //panda(7) + finger(2)

    for (int i=0; i<9; i++){ 
        joint_states.position[i] = msg.position[i];
        joint_states.velocity[i] = msg.velocity[i];
    }    

    joint_states_pub_.publish(joint_states);    
}

void ctrltypeCallback(const std_msgs::Int16ConstPtr &msg){
    ROS_WARN("%d", msg->data);
    
    if (msg->data != 899){
        int data = msg->data;
        ctrl_->ctrl_update(data);
    }
    else{
        if (isgrasp_)
            isgrasp_=false;
        else
            isgrasp_=true;
    }
}

void setRobotCommand(){
    robot_command_msg_.MODE = 1; //0:position control, 1:torque control
    robot_command_msg_.header.stamp = ros::Time::now();
    robot_command_msg_.time = time_;   
    
    for (int i=0; i<7; i++)
        robot_command_msg_.torque[i] = franka_torque_(i);    
}

void setGripperCommand(){
    if (isgrasp_){
        robot_command_msg_.torque[7] = -200.0;
        robot_command_msg_.torque[8] = -200.0;
    }
    else{
        robot_command_msg_.torque[7] = 100.0;
        robot_command_msg_.torque[8] = 100.0;
    }
}

void getEEState(){
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

void keyboard_event(){
    if (_kbhit()){
        int key;
        key = getchar();
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
                if (isgrasp_){
                    cout << "Release hand" << endl;
                    isgrasp_ = false;
                }
                else{
                    cout << "Grasp object" << endl;
                    isgrasp_ = true; 
                }
                break;
        }
    }
}

