from brax.robots.aliengo.utils import AliengoUtils
from brax.robots.aliengo.gait import AliengoGait, AliengoGaitParams
from brax.envs.base import RlwamEnv, State

from brax import actuator
from brax import kinematics
from brax.generalized.base import State as GeneralizedState
from brax.generalized import dynamics
from brax.generalized import integrator
from brax.generalized import mass
from brax import base
from brax.math import rotate, inv_rotate, quat_to_eulerzyx, eulerzyx_to_quat
from brax.generalized.pipeline import step as pipeline_step

from jax import numpy as jp
from typing import Optional, Any, Tuple, Callable
import jax
import flax


@flax.struct.dataclass
class ControlCommand:
    """Output of the low level controller which includes gait control and
    inverse kinematics. """
    q_des: jp.ndarray
    qd_des: jp.ndarray
    Kp: jp.ndarray
    Kd: jp.ndarray
    contact: jp.ndarray
    leg_phases: jp.ndarray
    pdes: jp.ndarray


class AliengoGoFast(RlwamEnv):
    """ Aliengo environment

    ### Observation space

    | Num | Observation                          | Min  | Max | Name           | Joint | Unit                 |
    | --- | ------------------------------------ | ---- | --- | -------------- | ----- | -------------------- |
    | 0   | w-quat of trunk                      | -1   | 1   | rw             | free  | angle (quat)         |
    | 1   | x-quat of trunk                      | -1   | 1   | rw             | free  | angle (quat)         |
    | 2   | y-quat of trunk                      | -1   | 1   | rw             | free  | angle (quat)         |
    | 3   | z-quat of trunk                      | -1   | 1   | rw             | free  | angle (quat)         |
    | 4   | front-right hip joint angle          | -Inf | Inf | FR_hip_joint   | hinge | angle (rad)          |
    | 5   | front-right thigh joint angle        | -Inf | Inf | FR_thigh_joint | hinge | angle (rad)          |
    | 6   | front-right calf joint angle         | -Inf | Inf | FR_calf_joint  | hinge | angle (rad)          |
    | 7   | front-left hip joint angle           | -Inf | Inf | FL_hip_joint   | hinge | angle (rad)          |
    | 8   | front-left thigh joint angle         | -Inf | Inf | FL_thigh_joint | hinge | angle (rad)          |
    | 9   | front-left calf joint angle          | -Inf | Inf | FL_calf_joint  | hinge | angle (rad)          |
    | 10  | rear-right hip joint angle           | -Inf | Inf | RR_hip_joint   | hinge | angle (rad)          |
    | 11  | rear-right thigh joint angle         | -Inf | Inf | RR_thigh_joint | hinge | angle (rad)          |
    | 12  | rear-right calf joint angle          | -Inf | Inf | RR_calf_joint  | hinge | angle (rad)          |
    | 13  | rear-left hip joint angle            | -Inf | Inf | RL_hip_joint   | hinge | angle (rad)          |
    | 14  | rear-left thigh joint angle          | -Inf | Inf | RL_thigh_joint | hinge | angle (rad)          |
    | 15  | rear-left calf joint angle           | -Inf | Inf | RL_calf_joint  | hinge | angle (rad)          |
    | 16  | x-velocity of trunk (body frame)     | -Inf | Inf | vx             | free  | velocity (m/s)       |
    | 17  | y-velocity of trunk (body frame)     | -Inf | Inf | vy             | free  | velocity (m/s)       |
    | 18  | z-velocity of trunk (body frame)     | -Inf | Inf | vz             | free  | velocity (m/s)       |
    | 19  | x-ang-velocity of trunk (body frame) | -Inf | Inf | wx             | free  | ang-velocity (rad/s) |
    | 20  | y-ang-velocity of trunk (body frame) | -Inf | Inf | wy             | free  | ang-velocity (rad/s) |
    | 21  | z-ang-velocity of trunk (body frame) | -Inf | Inf | wz             | free  | ang-velocity (rad/s) |
    | 22  | front-right hip joint speed          | -Inf | Inf | FR_hip_speed   | hinge | ang-speed (rad/s)    |
    | 23  | front-right thigh joint speed        | -Inf | Inf | FR_thigh_speed | hinge | ang-speed (rad/s)    |
    | 24  | front-right calf joint speed         | -Inf | Inf | FR_calf_speed  | hinge | ang-speed (rad/s)    |
    | 25  | front-left hip joint speed           | -Inf | Inf | FL_hip_speed   | hinge | ang-speed (rad/s)    |
    | 26  | front-left thigh joint speed         | -Inf | Inf | FL_thigh_speed | hinge | ang-speed (rad/s)    |
    | 27  | front-left calf joint speed          | -Inf | Inf | FL_calf_speed  | hinge | ang-speed (rad/s)    |
    | 28  | rear-right hip joint speed           | -Inf | Inf | RR_hip_speed   | hinge | ang-speed (rad/s)    |
    | 29  | rear-right thigh joint speed         | -Inf | Inf | RR_thigh_speed | hinge | ang-speed (rad/s)    |
    | 30  | rear-right calf joint speed          | -Inf | Inf | RR_calf_speed  | hinge | ang-speed (rad/s)    |
    | 31  | rear-left hip joint speed            | -Inf | Inf | RL_hip_speed   | hinge | ang-speed (rad/s)    |
    | 32  | rear-left thigh joint speed          | -Inf | Inf | RL_thigh_speed | hinge | ang-speed (rad/s)    |
    | 33  | rear-left calf joint speed           | -Inf | Inf | RL_calf_speed  | hinge | ang-speed (rad/s)    |
    | 34  | cos(phase)                           | -1   | 1   | cos_phase      | none  | unitless             |
    | 35  | sin(phase)                           | -1   | 1   | sin_phase      | none  | unitless             |

    ### Action space
    | Num   | Action                           | Min | Max |
    | ----- | -------------------------------- | --- | --- |
    | 0:4   | x foot position deltas           |     |     |
    | 4:8   | y foot position deltas           |     |     |
    | 8     | Body height, delta from standing |     |     |
    | 9:21  | P gains (if enabled)             |     |     |
    | 21:33 | D gains (if enabled)             |     |     |


    ### Actuator space

    | Num | Actuator                       | Min  | Max | Name     | Joint | Unit         |
    | --- | ------------------------------ | ---- | --- | -------- | ----- | ------------ |
    | 0   | front-right hip joint torque   | -Inf | Inf | FR_hip   | hinge | torque (N*m) |
    | 1   | front-right thigh joint torque | -Inf | Inf | FR_thigh | hinge | torque (N*m) |
    | 2   | front-right calf joint torque  | -Inf | Inf | FR_calf  | hinge | torque (N*m) |
    | 3   | front-left hip joint torque    | -Inf | Inf | FL_hip   | hinge | torque (N*m) |
    | 4   | front-left thigh joint torque  | -Inf | Inf | FL_thigh | hinge | torque (N*m) |
    | 5   | front-left calf joint torque   | -Inf | Inf | FL_calf  | hinge | torque (N*m) |
    | 6   | rear-right hip joint torque    | -Inf | Inf | RR_hip   | hinge | torque (N*m) |
    | 7   | rear-right thigh joint torque  | -Inf | Inf | RR_thigh | hinge | torque (N*m) |
    | 8   | rear-right calf joint torque   | -Inf | Inf | RR_calf  | hinge | torque (N*m) |
    | 9   | rear-left hip joint torque     | -Inf | Inf | RL_hip   | hinge | torque (N*m) |
    | 10  | rear-left thigh joint torque   | -Inf | Inf | RL_thigh | hinge | torque (N*m) |
    | 11  | rear-left calf joint torque    | -Inf | Inf | RL_calf  | hinge | torque (N*m) |
    """

    def __init__(
        self,
        policy_repeat=4,
        forward_cmd_vel_type='constant',  # 'constant' or 'sine'
        forward_cmd_vel_range=(0.0, 0.0),  # для текущей реализации используется среднее значение этого диапазона
        forward_cmd_vel_period_range=(5.0, 10.0),  # только для 'sine'
        turn_cmd_rate_range=(-jp.pi/8, jp.pi/8),
        initial_yaw_range=(-0.0, 0.0),
        contact_time_const=0.02,
        contact_time_const_range=None,
        contact_damping_ratio=1.0,
        friction_range=(0.6, 0.6),
        ground_roll_range=(0.0, 0.0),
        ground_pitch_range=(0.0, 0.0),
        joint_damping_perc_range=(1.0, 1.0),
        joint_gain_range=(1.0, 1.0),
        link_mass_perc_range=(1.0, 1.0),
        forward_vel_rew_weight=1.0,
        turn_rew_weight=0.5,
        pitch_rew_weight=0.20,
        roll_rew_weight=0.25,
        yaw_rew_weight=0.00,
        side_motion_rew_weight=0.25,
        z_vel_change_rew_weight=0.0,
        ang_vel_rew_weight=0.00,
        ang_change_rew_weight=0.25,
        joint_lim_rew_weight=0.15,
        torque_lim_rew_weight=0.15,
        joint_acc_rew_weight=0.05,
        action_rew_weight=0.1,
        cosmetic_rew_weight=0.0,
        energy_rew_weight=0.0,
        foot_z_rew_weight=0.0,
        torque_lim_penalty_weight=1.0,
        fallen_roll=jp.pi/4,
        fallen_pitch=jp.pi/4,
        forces_in_q_coords=False,
        include_height_in_obs=False,
        body_height_in_action_space=True,
        gains_in_action_space=False,
        backend='generalized',
        reward_type='normalized',
        used_cached_systems=False,
        healthy_delta_radius=2.0,  # not used
        healthy_delta_yaw=1.57,  # not used
        **kwargs
    ):

        self.sim_dt = 1/400  # simulation dt; 400 Hz

        # determines high level policy freq; (1/sim_dt)/policy_repeat Hz
        self.policy_repeat = policy_repeat

        sys = AliengoUtils.get_system(used_cached_systems)
        sys = sys.replace(dt=self.sim_dt)

        # here we are using the fast sim_dt with the approximate system instead
        # of self.sim_dt * self.policy_repeat
        self._sys_approx = AliengoUtils.get_approx_system(used_cached_systems)
        self._sys_approx = self._sys_approx.replace(dt=self.sim_dt)

        # normally this is use by Brax as the number of times to step the
        # physics pipeline for each environment step. However we have
        # overwritten the pipline_step function with our own behaviour which
        # steps the physics self.policy_repeat times. So we set this to 1.
        n_frames = 1
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._period = 0.50  # period of the gait cycle (sec)
        self._forward_cmd_vel = jp.mean(jp.array(forward_cmd_vel_range))
        self._initial_yaw_range = initial_yaw_range
        if contact_time_const_range is None:
            self._contact_time_const_range = (contact_time_const,
                                              contact_time_const)
        else:
            self._contact_time_const_range = contact_time_const_range
        self._contact_damping_ratio = contact_damping_ratio
        self._friction_range = friction_range
        self._ground_roll_range = ground_roll_range
        self._ground_pitch_range = ground_pitch_range
        self._joint_damping_perc_range = joint_damping_perc_range
        self._joint_gain_range = joint_gain_range
        self._link_mass_perc_range = link_mass_perc_range
        self._fallen_roll = fallen_roll
        self._fallen_pitch = fallen_pitch
        self._include_height_in_obs = include_height_in_obs
        self._body_height_in_action_space = body_height_in_action_space
        self._gains_in_action_space = gains_in_action_space

        if reward_type == 'normalized':
            self._reward_fn = self._reward_normalized 