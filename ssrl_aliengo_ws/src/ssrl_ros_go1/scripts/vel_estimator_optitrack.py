#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, Point

class MocapProcessor:
    def __init__(self):
        rospy.init_node('mocap_processor_node')

        # Параметры фильтрации
        self.position_noise_threshold = 0.001  # м
        self.velocity_noise_threshold = 0.001  # м/с
        self.filter_alpha = 0.2

        # Состояние фильтра
        self.filtered_position = Point()
        self.filtered_velocity = Point()
        self.last_position = None
        self.last_time = None
        self.anchor_position = None

        # Подписка и публикация
        self.odom_sub = rospy.Subscriber('/mocap_node/Robot_1/Odom', Odometry, self.odom_callback)
        self.pose_pub = rospy.Publisher('/optitrack_mocap/robot1/pose', PoseStamped, queue_size=10)
        self.twist_pub = rospy.Publisher('/optitrack_mocap/robot1/twist', TwistStamped, queue_size=10)

    def low_pass_filter(self, current, prev, alpha):
        return current if prev is None else alpha * current + (1 - alpha) * prev

    def is_noise(self, value, threshold):
        return abs(value) < threshold

    def odom_callback(self, msg):
        pose_in_original = PoseStamped()
        pose_in_original.header = msg.header
        pose_in_original.pose = msg.pose.pose

        # Преобразуем из исходной системы в ENU вручную
        pos_original = pose_in_original.pose.position
        pos_enu = Point(
            x=-pos_original.z,  # z(исходная) = x(ENU)
            y=pos_original.x,   # x(исходная) = y(ENU)
            z=-pos_original.y   # y(исходная) = z(ENU)
        )

        if self.anchor_position is None:
            self.anchor_position = pos_enu
            self.filtered_position = Point()
            self.filtered_velocity = Point()
            return

        rel_pos = Point(
            x=pos_enu.x - self.anchor_position.x,
            y=pos_enu.y - self.anchor_position.y,
            z=pos_enu.z - self.anchor_position.z
        )

        self.filtered_position.x = self.low_pass_filter(rel_pos.x, self.filtered_position.x, self.filter_alpha)
        self.filtered_position.y = self.low_pass_filter(rel_pos.y, self.filtered_position.y, self.filter_alpha)
        self.filtered_position.z = self.low_pass_filter(rel_pos.z, self.filtered_position.z, self.filter_alpha)

        for axis in ['x', 'y', 'z']:
            if self.is_noise(getattr(self.filtered_position, axis), self.position_noise_threshold):
                setattr(self.filtered_position, axis, 0.0)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = msg.header.stamp
        pose_msg.header.frame_id = "enu"
        pose_msg.pose.position = self.filtered_position
        pose_msg.pose.orientation = pose_in_original.pose.orientation
        self.pose_pub.publish(pose_msg)

        if self.last_position and self.last_time:
            dt = (msg.header.stamp - self.last_time).to_sec()
            if dt > 1e-4:
                vel = Point(
                    x=(self.filtered_position.x - self.last_position.x) / dt,
                    y=(self.filtered_position.y - self.last_position.y) / dt,
                    z=(self.filtered_position.z - self.last_position.z) / dt
                )

                self.filtered_velocity.x = self.low_pass_filter(vel.x, self.filtered_velocity.x, self.filter_alpha)
                self.filtered_velocity.y = self.low_pass_filter(vel.y, self.filtered_velocity.y, self.filter_alpha)
                self.filtered_velocity.z = self.low_pass_filter(vel.z, self.filtered_velocity.z, self.filter_alpha)

                for axis in ['x', 'y', 'z']:
                    if self.is_noise(getattr(self.filtered_velocity, axis), self.velocity_noise_threshold):
                        setattr(self.filtered_velocity, axis, 0.0)

                twist_msg = TwistStamped()
                twist_msg.header.stamp = msg.header.stamp
                twist_msg.header.frame_id = "enu"
                twist_msg.twist.linear = self.filtered_velocity
                self.twist_pub.publish(twist_msg)

        self.last_position = Point(
            x=self.filtered_position.x,
            y=self.filtered_position.y,
            z=self.filtered_position.z
        )
        self.last_time = msg.header.stamp


if __name__ == '__main__':
    try:
        MocapProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
