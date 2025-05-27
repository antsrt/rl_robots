#include <ros/ros.h>
#include <unitree_legged_msgs/LowCmd.h>
#include <unitree_legged_msgs/LowState.h>
#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "convert.h"
#include <chrono>
#include <pthread.h>

using namespace UNITREE_LEGGED_SDK;

const int LOW_CMD_LENGTH = 610;
const int LOW_STATE_LENGTH = 771;

class Custom
{
public:
    UDP low_udp;

    LowCmd low_cmd = {0};
    LowState low_state = {0};

    ros::Publisher* pub_low_ptr;

public:
    Custom(ros::Publisher* pub_low)
        : low_udp(8082, "192.168.123.10", 8007, LOW_CMD_LENGTH, LOW_STATE_LENGTH),
          pub_low_ptr(pub_low)
    {
        low_udp.InitCmdData(low_cmd);
    }

    void lowUdpSend()
    {
        low_udp.SetSend(low_cmd);
        low_udp.Send();
    }

    void lowUdpRecv()
    {
        low_udp.Recv();
        low_udp.GetRecv(low_state);
        
        unitree_legged_msgs::LowState low_state_ros;
        low_state_ros = state2rosMsg(low_state);
        pub_low_ptr->publish(low_state_ros);
    }
};

// Добавляем глобальный указатель на Custom
Custom* custom_ptr = nullptr;

ros::Subscriber sub_low;
ros::Publisher pub_low;

long low_count = 0;

void lowCmdCallback(const unitree_legged_msgs::LowCmd::ConstPtr &msg)
{
    printf("lowCmdCallback is running !\t%ld\n", ::low_count++);
    custom_ptr->low_cmd = rosMsg2Cmd(*msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ros_udp");
    ros::NodeHandle nh;

    sub_low = nh.subscribe("low_cmd", 1, lowCmdCallback);
    pub_low = nh.advertise<unitree_legged_msgs::LowState>("low_state", 1);

    // Создаем экземпляр Custom с publisher
    Custom custom(&pub_low);
    custom_ptr = &custom;

    LoopFunc loop_udpSend("low_udp_send", 0.002, 3, boost::bind(&Custom::lowUdpSend, &custom));
    LoopFunc loop_udpRecv("low_udp_recv", 0.002, 3, boost::bind(&Custom::lowUdpRecv, &custom));

    loop_udpSend.start();
    loop_udpRecv.start();

    printf("LOWLEVEL is initialized\n");
    ros::spin();

    return 0;
}
