#!/usr/bin/env python3
from geometry_msgs.msg import TwistStamped, PoseStamped
from ssrl_ros_go1 import env_dict
from unitree_legged_msgs.msg import LowState
from ssrl_ros_go1_msgs.msg import Observation, QuadrupedCommand, QuadrupedState
from brax.math import inv_rotate, quat_to_eulerzyx
from omegaconf import DictConfig
from jax import numpy as jp
import jax
import rospy

class Observer:
    def __init__(self, cfg: DictConfig):
        self.rate = 250
        self.cfg = cfg
        rospy.init_node('observer')
        rospy.loginfo(f"Building {self.cfg.env} environment...")
        self.env = env_dict[self.cfg.env](used_cached_systems=True)

        # Состояния и оцененные величины
        self.low_state = None  # Заполнится из сообщения low_state
        self.linear_vel = jp.zeros((3,))  # Оценка линейной скорости (через интеграцию)
        self.angular_vel = jp.zeros((3,))  # Угловая скорость из IMU
        self.last_tick = None  # Для вычисления dt по low_state.tick

        self.qped_cmd = QuadrupedCommand()
        self.step_count = 0
        self.qped_state = 'off'

        # Определяем метод оценки скорости через параметр ROS
        velocity_estimation = rospy.get_param('observer/velocity_estimation', 'onboard')

        if velocity_estimation == 'vicon':
            self.vicon_pos = jp.zeros((3,))
            self.vicon_quat = jp.array([1, 0, 0, 0])
            self.inv_rotate_jit = jax.jit(inv_rotate)
            self.pose_sub = rospy.Subscriber('vrpn_client_node/quad/pose',
                                             PoseStamped, self.vicon_pose_callback, tcp_nodelay=True)
            self.vel_sub = rospy.Subscriber('vrpn_client_node/quad/twist',
                                            TwistStamped, self.vicon_twist_callback, tcp_nodelay=True)
        elif velocity_estimation == 'onboard':
            rospy.loginfo("Using onboard velocity estimation")
            # Подписываемся на топик low_state для onboard-оценки
            self.low_sub = rospy.Subscriber('low_state', LowState, self.low_callback, tcp_nodelay=True)
        else:
            raise ValueError('Invalid velocity estimation method')

        self.qped_cmd_sub = rospy.Subscriber('quadruped_command', QuadrupedCommand, self.qped_cmd_callback)
        self.qped_state_sub = rospy.Subscriber('quadruped_state', QuadrupedState, self.qped_state_callback)
        self.pub = rospy.Publisher('observation', Observation, queue_size=10)

    def low_callback(self, low_state: LowState):
        """
        Обработка сообщения low_state для onboard-оценки.
        Поля из low_state (согласно вашему C/C++ заголовку):
          - tick: время в микросекундах.
          - imu.accelerometer: ускорения [ax, ay, az] в м/с².
          - imu.gyroscope: угловые скорости [gx, gy, gz] в рад/с.
          - imu.quaternion: ориентация в виде кватерниона.
        """
        self.low_state = low_state

        # Вычисляем интервал времени (dt) на основе поля tick
        current_tick = low_state.tick
        if self.last_tick is not None:
            dt = (current_tick - self.last_tick) / 1e6  # перевод микросекунд в секунды
        else:
            dt = 1.0 / self.rate  # если нет предыдущего значения, используем период по умолчанию
        self.last_tick = current_tick

        # Извлекаем ускорение из IMU
        acc = jp.array(low_state.imu.accelerometer)
        # Учитываем, что в измерениях акселерометра ось Z содержит гравитацию (~9.81 м/с²)
        gravity = jp.array([0.0, 0.0, 9.81])
        acc_corrected = acc - gravity

        # Интегрируем ускорение для получения линейной скорости
        self.linear_vel = self.linear_vel + acc_corrected * dt

        # Угловая скорость берется напрямую из IMU
        self.angular_vel = jp.array(low_state.imu.gyroscope)

    def vicon_pose_callback(self, pose: PoseStamped):
        self.vicon_pos = jp.array([pose.pose.position.x,
                                   pose.pose.position.y,
                                   pose.pose.position.z])
        self.vicon_quat = jp.array([pose.pose.orientation.w,
                                    pose.pose.orientation.x,
                                    pose.pose.orientation.y,
                                    pose.pose.orientation.z])

    def vicon_twist_callback(self, twist: TwistStamped):
        vel_global = jp.array([twist.twist.linear.x,
                               twist.twist.linear.y,
                               twist.twist.linear.z])
        self.linear_vel = self.inv_rotate_jit(vel_global, self.vicon_quat)

    def qped_cmd_callback(self, qped_cmd: QuadrupedCommand):
        self.qped_cmd = qped_cmd

    def qped_state_callback(self, qped_state: QuadrupedState):
        self.qped_state = qped_state.state
        if self.qped_state in ['off', 'stand', 'standing_up']:
            self.step_count = 0

    def publish_observation(self):
        msg = Observation()

        # Извлекаем данные моторных состояний (используем первые 12 моторов)
        q = [ms.q for ms in self.low_state.motorState][:12]
        qd = [ms.dq for ms in self.low_state.motorState][:12]
        t = self.step_count / self.rate
        phase = 2 * jp.pi * t / self.env._period

        # Если используется окружение, где позиция базы требуется (например, круговое движение)
        if self.cfg.env in ['Go1GoFastCircle', 'Go1GoFastCircleJa']:
            # В onboard-режиме, если Vicon не используется, можно реализовать альтернативную оценку позиции,
            # но здесь оставляем тот же вызов (он будет работать только если Vicon-подписка активна)
            msg.observation[self.env._circle_pose_idxs] = circle_pose(
                self.env._radius, self.vicon_pos[0], self.vicon_pos[1], self.vicon_quat)
        elif self.cfg.env == 'Go1GoFast':
            # Для Go1GoFast ориентация берется из данных IMU
            msg.observation[self.env._quat_idxs] = self.low_state.imu.quaternion

        # Передаем оцененные скорости: линейную и угловую
        msg.observation[self.env._base_vel_idxs] = self.linear_vel
        msg.observation[self.env._rpy_rate_idxs] = self.angular_vel

        msg.observation[self.env._q_idxs] = q
        msg.observation[self.env._qd_idxs] = qd
        msg.observation[self.env._cos_phase_idx] = jp.cos(phase)
        msg.observation[self.env._sin_phase_idx] = jp.sin(phase)

        self.pub.publish(msg)

    def run(self):
        rospy.loginfo("Observer started")
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            # Убедимся, что low_state получено
            if self.low_state is not None:
                self.publish_observation()
            if self.qped_state == 'walk':
                self.step_count += 1
            rate.sleep()

def circle_pose(radius: float, x: float, y: float, quat: jp.ndarray):
    """
    Функция для вычисления относительной позиции базы для окружения Go1GoFastCircle.
    Здесь вычисляется отклонение радиуса и разница углов.
    """
    delta_radius = jp.sqrt(x**2 + y**2) - radius
    roll, pitch, yaw = quat_to_eulerzyx(quat)
    nominal_yaw = jp.arctan2(y, x) + jp.pi / 2
    delta_yaw = jp.arctan2(jp.sin(yaw - nominal_yaw), jp.cos(yaw - nominal_yaw))
    return jp.array([delta_radius, roll, pitch, delta_yaw])

if __name__ == '__main__':
    # Здесь предполагается, что конфигурация (cfg) передается извне (например, из YAML-файла)
    # или создается вручную. Для тестирования можно создать dummy-конфигурацию.
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "env": "Go1GoFast"  # или "Go1GoFastCircle" и т.д.
    })
    observer = Observer(cfg)
    observer.run()
