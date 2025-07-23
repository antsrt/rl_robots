#include <ros/ros.h>
#include <ssrl_ros_go1_msgs/PdTarget.h>
#include <ssrl_ros_go1_msgs/TorqueTarget.h>
#include <unitree_legged_msgs/LowCmd.h>
#include "unitree_legged_sdk/unitree_legged_sdk.h"

using namespace UNITREE_LEGGED_SDK;

class LowCmdPub {
public:
    LowCmdPub() : initiated_flag_(false), count_(0) {
        const auto queue_size = 1000;
        pd_cmd_sub_ = nh_.subscribe("pd_target", queue_size, &LowCmdPub::pdCallback, this, ros::TransportHints().tcpNoDelay(true));
        tq_cmd_sub_ = nh_.subscribe("torque_target", queue_size, &LowCmdPub::torqueCallback, this, ros::TransportHints().tcpNoDelay(true));
        cmd_pub_ = nh_.advertise<unitree_legged_msgs::LowCmd>("low_cmd", queue_size);

        lcmd_.head[0] = 0xFE;
        lcmd_.head[1] = 0xEF;
        lcmd_.levelFlag = LOWLEVEL;
        for (int i = 0; i < 12; i++) {
            lcmd_.motorCmd[i].mode = 0x0A;
            lcmd_.motorCmd[i].q = PosStopF;
            lcmd_.motorCmd[i].Kp = 0;
            lcmd_.motorCmd[i].dq = VelStopF;
            lcmd_.motorCmd[i].Kd = 0;
            lcmd_.motorCmd[i].tau = 0;
        }
    }

    void publish() {
        if (count_ < 10) {
            count_++;
            ROS_INFO("Waiting for initialization: %d/10", count_);
        } else {
            initiated_flag_ = true;
        }
        if (initiated_flag_) {
            cmd_pub_.publish(lcmd_);
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher cmd_pub_;
    ros::Subscriber pd_cmd_sub_;
    ros::Subscriber tq_cmd_sub_;
    unitree_legged_msgs::LowCmd lcmd_;
    bool initiated_flag_;
    int count_;

    void pdCallback(ssrl_ros_go1_msgs::PdTarget pd) {
        ROS_INFO("Received pd_target: mode=%d, q_des[0]=%f, Kp[0]=%f", pd.mode, pd.q_des[0], pd.Kp[0]);
        for (int i = 0; i < 12; i++) {
            lcmd_.motorCmd[i].mode = pd.mode;
            lcmd_.motorCmd[i].q = pd.q_des[i];
            lcmd_.motorCmd[i].Kp = pd.Kp[i];
            lcmd_.motorCmd[i].dq = pd.qd_des[i];
            lcmd_.motorCmd[i].Kd = pd.Kd[i];
            lcmd_.motorCmd[i].tau = 0;
        }
    }

    void torqueCallback(ssrl_ros_go1_msgs::TorqueTarget tq) {
        ROS_INFO("Received torque_target: mode=%d, tau_des[0]=%f", tq.mode, tq.tau_des[0]);
        for (int i = 0; i < 12; i++) {
            lcmd_.motorCmd[i].mode = tq.mode;
            lcmd_.motorCmd[i].q = PosStopF;
            lcmd_.motorCmd[i].Kp = 0;
            lcmd_.motorCmd[i].dq = VelStopF;
            lcmd_.motorCmd[i].Kd = 0;
            lcmd_.motorCmd[i].tau = tq.tau_des[i];
        }
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "low_cmd_pub");
    LowCmdPub lcp;
    ros::Rate loop_rate(500); // 500 Hz
    ROS_INFO_STREAM("Started low_cmd publisher");
    while (ros::ok()) {
        ros::spinOnce();
        lcp.publish(); // Публикуем lcmd_ с частотой 500 Гц
        loop_rate.sleep();
    }
    return 0;
}