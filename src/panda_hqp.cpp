#include "kimm_phri_panda/panda_hqp.h"

using namespace pinocchio;
using namespace Eigen;
using namespace std;
using namespace kimmhqp;
using namespace kimmhqp::trajectory;
using namespace kimmhqp::math;
using namespace kimmhqp::tasks;
using namespace kimmhqp::solver;
using namespace kimmhqp::robot;
using namespace kimmhqp::contacts;

namespace RobotController{
    // FrankaWrapper::FrankaWrapper(const std::string & robot_node, const bool & issimulation, ros::NodeHandle & node)
    FrankaWrapper::FrankaWrapper(const std::string & robot_node, const bool & issimulation, ros::NodeHandle & node, const int & ctrl_mode)
    : robot_node_(robot_node), issimulation_(issimulation), n_node_(node)
    {
        time_ = 0.;        
        node_index_ = 0;
        cnt_ = 0;

        // mode_change_ = false;
        // ctrl_mode_ = 0;

        ctrl_mode_ = ctrl_mode;
        if (ctrl_mode_ == 1)
        {
            mode_change_ = true;
        }
        else //ctrl_mode = 0
        {
            mode_change_ = false;
        }                
    }

    void FrankaWrapper::initialize(){
        // Robot for pinocchio
        string model_path, urdf_name;
        n_node_.getParam("/" + robot_node_ +"/robot_urdf_path", model_path);
        n_node_.getParam("/" + robot_node_ +"/robot_urdf", urdf_name);        //"panda_arm_hand_l.urdf"

        vector<string> package_dirs;
        package_dirs.push_back(model_path);
        string urdfFileName = package_dirs[0] + urdf_name;
        robot_ = std::make_shared<RobotWrapper>(urdfFileName, package_dirs, false, false); //first false : w/o mobile, true : w/ mobile
        model_ = robot_->model();
        
        //nq_/nv_/na_ is # of joint w.r.t pinocchio model ("panda_arm_hand_l.urdf"), so there is no gripper joints
        nq_ = robot_->nq(); //7 : franka (7) 
        nv_ = robot_->nv(); //7
        na_ = robot_->na(); //7 

        // State (for pinocchio)
        state_.q_.setZero(nq_);
        state_.v_.setZero(nv_);
        state_.dv_.setZero(nv_);
        state_.torque_.setZero(na_);
        state_.tau_.setZero(na_);

        // tsid
        tsid_ = std::make_shared<InverseDynamicsFormulationAccForce>("tsid", *robot_);
        tsid_->computeProblemData(time_, state_.q_, state_.v_);
        data_ = tsid_->data();

        // tasks
        postureTask_ = std::make_shared<TaskJointPosture>("task-posture", *robot_);
        VectorXd posture_gain(na_);
        if (!issimulation_) //for real
        	// posture_gain << 200., 200., 200., 200., 200., 200., 200.;
            // posture_gain << 250., 150., 250., 100., 100., 100., 100.;
            // posture_gain << 100., 100., 100., 100., 100., 100., 100.;            
            posture_gain << 100., 100., 100., 200., 200., 200., 200.;
        else // for simulation
        	// posture_gain << 4000., 4000., 4000., 4000., 4000., 4000., 4000.;
            posture_gain << 40000., 40000., 40000., 40000., 40000., 40000., 40000.;

        postureTask_->Kp(posture_gain);
        postureTask_->Kd(2.0*postureTask_->Kp().cwiseSqrt());

        Vector3d ee_offset(0.0, 0, 0.0); //0.165
        ee_offset_ << ee_offset;

        Adj_mat.resize(6,6);
        Adj_mat.setIdentity();
        Adj_mat(0, 4) = -ee_offset_(2);
        Adj_mat(0, 5) = ee_offset_(1);
        Adj_mat(1, 3) = ee_offset_(2);
        Adj_mat(1, 5) = -ee_offset_(0);
        Adj_mat(2, 3) = -ee_offset_(1);
        Adj_mat(2, 4) = ee_offset_(0);

        VectorXd ee_gain(6);
        ee_gain << 100., 100., 100., 400., 400., 600.;
        // ee_gain << 100., 100., 100., 200., 200., 200.;

        eeTask_ = std::make_shared<TaskSE3Equality>("task-se3", *robot_, "panda_joint7", ee_offset);
        eeTask_->Kp(ee_gain*Vector::Ones(6));
        eeTask_->Kd(2.0*eeTask_->Kp().cwiseSqrt());

        torqueBoundsTask_ = std::make_shared<TaskJointBounds>("task-torque-bounds", *robot_);
        Vector dq_max = 500000.0*Vector::Ones(na_);
        dq_max(0) = 500.; //? 
        dq_max(1) = 500.; //?
        Vector dq_min = -dq_max;
        torqueBoundsTask_->setJointBounds(dq_min, dq_max);        

        // trajecotries
        sampleEE_.resize(12, 6); //12=3(translation)+9(rotation matrix), 6=3(translation)+3(rotation)
        samplePosture_.resize(na_); //na_=7 franka 7

        trajPosture_Cubic_ = std::make_shared<TrajectoryEuclidianCubic>("traj_posture");
        trajPosture_Constant_ = std::make_shared<TrajectoryEuclidianConstant>("traj_posture_constant");
        trajPosture_Timeopt_ = std::make_shared<TrajectoryEuclidianTimeopt>("traj_posture_timeopt");

        trajEE_Cubic_ = std::make_shared<TrajectorySE3Cubic>("traj_ee");
        trajEE_Constant_ = std::make_shared<TrajectorySE3Constant>("traj_ee_constant");
        Vector3d Maxvel_ee = Vector3d::Ones()*0.2;
        Vector3d Maxacc_ee = Vector3d::Ones()*0.2;
        trajEE_Timeopt_ = std::make_shared<TrajectorySE3Timeopt>("traj_ee_timeopt", Maxvel_ee, Maxacc_ee);

        // solver
        solver_ = SolverHQPFactory::createNewSolver(SOLVER_HQP_QPOASES, "qpoases");

        // service
        reset_control_ = true;
        planner_res_ = false;
    }

    void FrankaWrapper::franka_update(const sensor_msgs::JointState& msg){ //for simulation (mujoco)
        // mujoco callback msg
        // msg.position : 7(joint) + 2(gripper)
        // msg.velocity : 7(joint) + 2(gripper)

        assert(issimulation_);
        for (int i=0; i< nq_; i++){
            state_.q_(i) = msg.position[i];
            state_.v_(i) = msg.velocity[i];
        }
    }
    void FrankaWrapper::franka_update(const Vector7d& q, const Vector7d& qdot){ //for experiment
        assert(!issimulation_);
        state_.q_.tail(nq_) = q;
        state_.v_.tail(nq_) = qdot;
    }        

    void FrankaWrapper::franka_update(const Vector7d& q, const Vector7d& qdot, const Vector7d& tau){ //for experiment, use pinocchio::aba
        assert(!issimulation_);
        state_.q_.tail(nq_) = q;
        state_.v_.tail(nv_) = qdot;

        state_.tau_.tail(na_) = tau;
    }        

    void FrankaWrapper::ctrl_update(const int& msg){
        ctrl_mode_ = msg;
        ROS_INFO("[ctrltypeCallback] %d", ctrl_mode_);
        mode_change_ = true;
    }

    void FrankaWrapper::compute(const double& time){
        time_ = time;

        robot_->computeAllTerms(data_, state_.q_, state_.v_);
        // robot_->computeAllTerms_ABA(data_, state_.q_, state_.v_, state_.tau_);

        if (ctrl_mode_ == 0){ //g // gravity mode
            state_.torque_.setZero();
        }
        if (ctrl_mode_ == 1){ //h //init position
            if (mode_change_){                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                tsid_->addMotionTask(*postureTask_, 1e-2, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);

                q_ref_.setZero(7);
                q_ref_(0) =  0;//M_PI /4.0;
                q_ref_(1) = 0.0 * M_PI / 180.0;
                q_ref_(3) = -M_PI / 2.0;
                q_ref_(5) = M_PI/ 2.0;
                q_ref_(6) = -M_PI/ 4.0;

                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(q_ref_);

                reset_control_ = false;
                mode_change_ = false;

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7"));
                // cout << H_ee_ref_ << endl;
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);

            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));

        }        
        if (ctrl_mode_ == 2){ //a //approach to object
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-16, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //posture (try to maintain current joint configuration)
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(2.0);
                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7"));

                SE3 T_offset;
                T_offset.setIdentity();
                T_offset.translation(ee_offset_);
                H_ee_ref_ = H_ee_ref_ * T_offset;

                trajEE_Cubic_->setInitSample(H_ee_ref_);
                H_ee_ref_.translation()(0) += 0.1;                
                trajEE_Cubic_->setGoalSample(H_ee_ref_);

                reset_control_ = false;
                mode_change_ = false;
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));

        }
        if (ctrl_mode_ == 3){ //s //home and align base and joint7 coordinate
            if (mode_change_){     
                //remove                           
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-6, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);

                q_ref_.setZero(7);
                q_ref_(0) =  0;//M_PI /4.0;
                q_ref_(1) = 0.0 * M_PI / 180.0;
                q_ref_(3) = - M_PI / 2.0;
                q_ref_(5) = M_PI/ 2.0;
                // q_ref_(6) = -M_PI/ 4.0;
                q_ref_(6) = 0.0;

                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(q_ref_);

                reset_control_ = false;
                mode_change_ = false;

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7"));
                // cout << H_ee_ref_ << endl;
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);

            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));
        }
        if (ctrl_mode_ == 4){ //d //rotate ee for object estimation
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-6, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);                

                //posture
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(q_ref_);

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(2.0);
                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7"));
              
                Vector3d ee_offset;
                ee_offset << 0.0, 0.0, 0.2; //joint7 to ee

                SE3 T_offset;
                T_offset.setIdentity();
                T_offset.translation(ee_offset);
                // H_ee_ref_ = H_ee_ref_ * T_offset;

                trajEE_Cubic_->setInitSample(H_ee_ref_);

                Matrix3d Rx, Ry, Rz;
                Rx.setIdentity();
                Ry.setIdentity();
                Rz.setIdentity();
                double angle = 15.0*M_PI/180.0;
                double anglex = -angle;
                double angley = angle;
                rotx(anglex, Rx);
                roty(angley, Ry);
                // rotz(angle, Rz);
                // H_ee_ref_.rotation() = H_ee_ref_.rotation()*Rx;
                // H_ee_ref_.rotation() = Rz * Ry * Rx * H_ee_ref_.rotation();

                Vector3d trans_offset;
                trans_offset << -ee_offset(2)*sin(angle), -ee_offset(2)*sin(angle), ee_offset(2)*(1.0-cos(angle));
                // H_ee_ref_.translation() = H_ee_ref_.translation() + H_ee_ref_.rotation()  * ee_offset;
                H_ee_ref_.translation() = H_ee_ref_.translation() + H_ee_ref_.rotation() * trans_offset;

                H_ee_ref_.rotation() = H_ee_ref_.rotation()*Rx*Ry;
                
                
                trajEE_Cubic_->setGoalSample(H_ee_ref_);                

                reset_control_ = false;
                mode_change_ = false;
            }
           
            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));
        }

        if (ctrl_mode_ == 5){ //f //rotate ee with sine motion for object estimation
            if (mode_change_){
                //remove                
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-6, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //posture
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(q_ref_);

                //ee
                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(2.0);
                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7"));

                SE3 T_offset;
                T_offset.setIdentity();
                T_offset.translation(ee_offset_);
                H_ee_ref_ = H_ee_ref_ * T_offset;

                trajEE_Cubic_->setInitSample(H_ee_ref_);
                trajEE_Cubic_->setGoalSample(H_ee_ref_);
               
                reset_control_ = false;
                mode_change_ = false;

                est_time = time_;
            }       

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            //////////////////
            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext(); //TrajectorySample
            // eeTask_->setReference(sampleEE_); //TaskSE3Equality

            TrajectorySample m_sample;
            m_sample.resize(12, 6); //pos 12, vel 6, pos : translation 3 + rotation matrix 9
            m_sample = sampleEE_;   //take current pos

            if (time_ - est_time < 10.0){
                Matrix3d Rx, Ry, Rz;
                Rx.setIdentity();
                Ry.setIdentity();
                Rz.setIdentity();
                double anglex = 10*M_PI/180.0*sin(1*M_PI*time_);
                double angley = 10*M_PI/180.0*sin(2*M_PI*time_);
                double anglez = 30*M_PI/180.0*sin(0.2*M_PI*time_);
                // rotx(anglex, Rx);
                // roty(angley, Ry);
                rotz(anglez, Rz);

                pinocchio::SE3 H_EE_ref_estimation;
                H_EE_ref_estimation = H_ee_ref_;
                H_EE_ref_estimation.rotation() = Rz * Ry * Rx * H_ee_ref_.rotation();

                SE3ToVector(H_EE_ref_estimation, m_sample.pos);
            }

            eeTask_->setReference(m_sample);
            //////////////////

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));
        }      

        if (ctrl_mode_ == 20){ //t //F_ext test
            if (mode_change_){
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //tsid_->addMotionTask(*postureTask_, 1e-2, 1);
                // tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1, 0);

                // trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                // trajPosture_Cubic_->setDuration(1.0);
                // trajPosture_Cubic_->setStartTime(time_);
                // trajPosture_Cubic_->setGoalSample(q_ref_);    

                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7"));     
                trajEE_Cubic_->setInitSample(H_ee_ref_);
                trajEE_Cubic_->setDuration(3.0);
                trajEE_Cubic_->setStartTime(time_);                                
                trajEE_Cubic_->setGoalSample(H_ee_ref_);
               
                // q_ref_ = state_.q_;       

                reset_control_ = false;
                mode_change_ = false;     
            }
            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();    
              
            sampleEE_.pos.head(2) = robot_->position(data_, robot_->model().getJointId("panda_joint7")).translation().head(2);               
            
            eeTask_->setReference(sampleEE_);                        

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);       
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));
        }

        if (ctrl_mode_ == 30){ //q //collaboration work with robot arm
            if (mode_change_){
            //remove                
            tsid_->removeTask("task-se3");
            tsid_->removeTask("task-posture");
            tsid_->removeTask("task-torque-bounds");

            //add
            tsid_->addMotionTask(*postureTask_, 1e-16, 1);
            tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
            tsid_->addMotionTask(*eeTask_, 1.0, 0);

            //posture (try to maintain current joint configuration)
            trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
            trajPosture_Cubic_->setDuration(2.0);
            trajPosture_Cubic_->setStartTime(time_);
            trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

            //ee
            trajEE_Cubic_->setStartTime(time_);
            trajEE_Cubic_->setDuration(2.0);
            H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7"));

            SE3 T_offset;
            T_offset.setIdentity();
            T_offset.translation(ee_offset_);
            H_ee_ref_ = H_ee_ref_ * T_offset;

            trajEE_Cubic_->setInitSample(H_ee_ref_);
            H_ee_ref_.translation()(0) -= 0.3;                
            trajEE_Cubic_->setGoalSample(H_ee_ref_);

            reset_control_ = false;
            mode_change_ = false;
        }

        trajPosture_Cubic_->setCurrentTime(time_);
        samplePosture_ = trajPosture_Cubic_->computeNext();
        postureTask_->setReference(samplePosture_);

        trajEE_Cubic_->setCurrentTime(time_);
        sampleEE_ = trajEE_Cubic_->computeNext();
        eeTask_->setReference(sampleEE_);

        const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
        state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));
        }

        if (ctrl_mode_ == 99){ //p //print current ee state
            if (mode_change_){
                //remove               
                tsid_->removeTask("task-se3");
                tsid_->removeTask("task-posture");
                tsid_->removeTask("task-torque-bounds");

                //add
                tsid_->addMotionTask(*postureTask_, 1e-6, 1);
                tsid_->addMotionTask(*torqueBoundsTask_, 1.0, 0);
                tsid_->addMotionTask(*eeTask_, 1.0, 0);

                //traj
                trajPosture_Cubic_->setInitSample(state_.q_.tail(na_));
                trajPosture_Cubic_->setDuration(2.0);
                trajPosture_Cubic_->setStartTime(time_);
                trajPosture_Cubic_->setGoalSample(state_.q_.tail(na_));

                trajEE_Cubic_->setStartTime(time_);
                trajEE_Cubic_->setDuration(2.0);
                H_ee_ref_ = robot_->position(data_, robot_->model().getJointId("panda_joint7"));
                trajEE_Cubic_->setInitSample(H_ee_ref_);
                trajEE_Cubic_->setGoalSample(H_ee_ref_);

                cout << H_ee_ref_ << endl;

                reset_control_ = false;
                mode_change_ = false;
            }

            trajPosture_Cubic_->setCurrentTime(time_);
            samplePosture_ = trajPosture_Cubic_->computeNext();
            postureTask_->setReference(samplePosture_);

            trajEE_Cubic_->setCurrentTime(time_);
            sampleEE_ = trajEE_Cubic_->computeNext();
            eeTask_->setReference(sampleEE_);

            const HQPData & HQPData = tsid_->computeProblemData(time_, state_.q_, state_.v_);
            state_.torque_ = tsid_->getAccelerations(solver_->solve(HQPData));
        }
    }

    void FrankaWrapper::franka_output(VectorXd & qacc) {
        qacc = state_.torque_.tail(na_);
    }    

    void FrankaWrapper::com(Eigen::Vector3d & com){
        //The element com[0] corresponds to the center of mass position of the whole model and expressed in the global frame.
        com = robot_->com(data_);
    }

    void FrankaWrapper::position(pinocchio::SE3 & oMi){
        oMi = robot_->position(data_, robot_->model().getJointId("panda_joint7"));
    }

    void FrankaWrapper::velocity(pinocchio::Motion & vel){
        // vel = robot_->velocity(data_, robot_->model().getJointId("panda_joint7"));

        Motion v_frame;
        SE3 T_offset;
        T_offset.setIdentity();
        T_offset.translation(ee_offset_);

        robot_->frameVelocity(data_, robot_->model().getFrameId("panda_joint7"), v_frame);

        vel = T_offset.act(v_frame); //if T_offset is identity, then T_offset.act(v_frame) = data.v[robot_->model().getJointId("panda_joint7")]

        // SE3 m_wMl_prev;
        // robot_->framePosition(data_, robot_->model().getFrameId("panda_joint7"), m_wMl_prev);
        // std::cout << m_wMl_prev << std::endl;
    }

    void FrankaWrapper::velocity_origin(pinocchio::Motion & vel){
        vel = robot_->velocity_origin(data_, robot_->model().getJointId("panda_joint7"));
    }

    void FrankaWrapper::velocity_global(pinocchio::Motion & vel){
        SE3 m_wMl_prev, m_wMl;
        Motion v_frame;

        SE3 T_offset;
        T_offset.setIdentity();
        T_offset.translation(ee_offset_);

        robot_->framePosition(data_, robot_->model().getFrameId("panda_joint7"), m_wMl_prev);
        m_wMl = m_wMl_prev * T_offset;

        robot_->frameVelocity(data_, robot_->model().getFrameId("panda_joint7"), v_frame);

        vel = m_wMl.act(v_frame);
    }

    void FrankaWrapper::acceleration(pinocchio::Motion & accel){        
        // accel = robot_->acceleration(data_, robot_->model().getJointId("panda_joint7"));                        
        
        Motion a_frame, m_drift;
        SE3 T_offset;
        T_offset.setIdentity();
        T_offset.translation(ee_offset_);

        robot_->frameAcceleration(data_, robot_->model().getFrameId("panda_joint7"), a_frame);
        robot_->frameClassicAcceleration(data_, robot_->model().getFrameId("panda_joint7"), m_drift);
        
        // accel = T_offset.act(a_frame+m_drift);
        // accel = T_offset.act(m_drift);
        accel = T_offset.act(a_frame);
    }

    void FrankaWrapper::acceleration_origin(pinocchio::Motion & accel){
        accel = robot_->acceleration_origin(data_, robot_->model().getJointId("panda_joint7"));
    }

    void FrankaWrapper::acceleration_global(pinocchio::Motion & accel){
        SE3 m_wMl_prev, m_wMl;
        Motion a_frame, m_drift;

        SE3 T_offset;
        T_offset.setIdentity();
        T_offset.translation(ee_offset_);

        robot_->framePosition(data_, robot_->model().getFrameId("panda_joint7"), m_wMl_prev);
        m_wMl = m_wMl_prev * T_offset;

        robot_->frameAcceleration(data_, robot_->model().getFrameId("panda_joint7"), a_frame);
        robot_->frameClassicAcceleration(data_, robot_->model().getFrameId("panda_joint7"), m_drift);

        // accel = m_wMl.act(a_frame+m_drift);
        accel = m_wMl.act(a_frame);
    }

    void FrankaWrapper::force(pinocchio::Force & force){
        force = robot_->force(data_, robot_->model().getJointId("panda_joint7"));
    }

    void FrankaWrapper::force_origin(pinocchio::Force & force){
        force = robot_->force_origin(data_, robot_->model().getJointId("panda_joint7"));
    }

    void FrankaWrapper::force_global(pinocchio::Force & force){
        SE3 m_wMl;
        Force f_frame;
        robot_->framePosition(data_, robot_->model().getFrameId("panda_joint7"), m_wMl);
        robot_->frameForce(data_, robot_->model().getFrameId("panda_joint7"), f_frame);
        force = m_wMl.act(f_frame);
    }

    void FrankaWrapper::tau(VectorXd & tau_vec){
        tau_vec = robot_->jointTorques(data_).tail(na_);
    }

    void FrankaWrapper::ddq(VectorXd & ddq_vec){
        ddq_vec = robot_->jointAcceleration(data_).tail(na_);
    }

    void FrankaWrapper::mass(MatrixXd & mass_mat){
        mass_mat = robot_->mass(data_).bottomRightCorner(na_, na_);
    }

    void FrankaWrapper::nle(VectorXd & nle_vec){
        nle_vec = robot_->nonLinearEffects(data_).tail(na_);
    }

    void FrankaWrapper::g(VectorXd & g_vec){
        g_vec = data_.g.tail(na_);
    }

    void FrankaWrapper::g_joint7(VectorXd & g_vec){
        Vector3d g_global;
        // g_global << 0.0, 0.0, -9.81;
        g_global << 0.0, 0.0, 9.81;
        
        SE3 m_wMl;        
        robot_->framePosition(data_, robot_->model().getFrameId("panda_joint7"), m_wMl);
        m_wMl.translation() << 0.0, 0.0, 0.0; //transform only with rotation
        g_vec = m_wMl.actInv(g_global);
    }

    void FrankaWrapper::JWorld(MatrixXd & Jo){
        Data::Matrix6x Jo2;
        // Jo2.resize(6, 10);
        Jo2.resize(6, robot_->nv());
        robot_->jacobianWorld(data_, robot_->model().getJointId("panda_joint7"), Jo2);
        Jo = Jo2.bottomRightCorner(6, 7);
    }

    void FrankaWrapper::JLocal_offset(MatrixXd & Jo){
        Data::Matrix6x Jo2;
        // Jo2.resize(6, 10);
        Jo2.resize(6, robot_->nv());
        robot_->frameJacobianLocal(data_, robot_->model().getFrameId("panda_joint7"), Jo2);
        Jo2 = Adj_mat * Jo2;
        Jo = Jo2.bottomRightCorner(6, 7);
    }

    void FrankaWrapper::dJLocal_offset(MatrixXd & dJo){
        Data::Matrix6x dJo2;
        // Jo2.resize(6, 10);
        dJo2.resize(6, robot_->nv());
        robot_->frameJacobianTimeVariationLocal(data_, robot_->model().getFrameId("panda_joint7"), dJo2);
        dJo2 = Adj_mat * dJo2;
        dJo = dJo2.bottomRightCorner(6, 7);
    }

    void FrankaWrapper::ee_state(Vector3d & pos, Eigen::Quaterniond & quat){
        for (int i=0; i<3; i++)
            pos(i) = robot_->position(data_, robot_->model().getJointId("panda_joint7")).translation()(i);

        Quaternion<double> q(robot_->position(data_, robot_->model().getJointId("panda_joint7")).rotation());
        quat = q;
    }

    void FrankaWrapper::rotx(double & angle, Eigen::Matrix3d & rot){
        rot.row(0) << 1,           0,           0;
        rot.row(1) << 0,           cos(angle), -sin(angle);
        rot.row(2) << 0,           sin(angle),  cos(angle);
    }

    void FrankaWrapper::roty(double & angle, Eigen::Matrix3d & rot){
        rot.row(0) <<  cos(angle),           0,            sin(angle);
        rot.row(1) <<  0,                    1,            0 ;
        rot.row(2) << -sin(angle),           0,            cos(angle);
    }

    void FrankaWrapper::rotz(double & angle, Eigen::Matrix3d & rot){
        rot.row(0) << cos(angle), -sin(angle),  0;
        rot.row(1) << sin(angle),  cos(angle),  0;
        rot.row(2) << 0,                    0,  1;
    }
}// namespace
