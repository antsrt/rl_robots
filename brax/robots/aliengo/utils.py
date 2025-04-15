from brax.base import System
from etils import epath
from brax.io import mjcf
from jax import numpy as jp
import jax
import dill
from pathlib import Path


class AliengoUtils:
    """Utility functions for the Aliengo."""

    """
    Properties
    """
    THIGH_OFFSET = 0.083  # константа: длина выноса бедра (из XML)
    """constant: the length of the thigh motor"""

    LEG_OFFSET_X = 0.2399  # расстояние по x от центра масс до основания ноги (из XML)
    """constant: x distance from the robot COM to the leg base."""

    LEG_OFFSET_Y = 0.051  # расстояние по y от центра масс до основания ноги (из XML)
    """constant: y distance from the robot COM to the leg base."""

    THIGH_LENGTH = 0.25  # длина бедра (из XML)
    """constant: length of the thigh link"""

    CALF_LENGTH = 0.25  # длина голени (из XML)
    """constant: length of the calf link"""

    STANDING_FOOT_POSITIONS = jp.array([
        0.2399, -0.135, -0.335,
        0.2399, 0.135, -0.335,
        -0.2399, -0.135, -0.335,
        -0.2399, 0.135, -0.335
    ])

    # Углы суставов в стоячем положении для каждой ноги
    # Значения примерные и могут требовать калибровки
    STANDING_JOINT_ANGLES_FR = jp.array([-0.1, 0.8, -1.5])
    STANDING_JOINT_ANGLES_FL = jp.array([0.1, 0.8, -1.5])
    STANDING_JOINT_ANGLES_RR = jp.array([-0.1, 0.8, -1.5])
    STANDING_JOINT_ANGLES_RL = jp.array([0.1, 0.8, -1.5])

    ALL_STANDING_JOINT_ANGLES = jp.concatenate([
        STANDING_JOINT_ANGLES_FR,
        STANDING_JOINT_ANGLES_FL,
        STANDING_JOINT_ANGLES_RR,
        STANDING_JOINT_ANGLES_RL
    ])

    JOINT_LIMIT_PAD = 0.1
    """constant: the amount to pad the joint limits"""

    # Ограничения углов суставов с отступом
    # Значения из aliengo_const.h
    LOWER_JOINT_LIMITS = jp.array([-0.873, -0.524, -2.775]) + JOINT_LIMIT_PAD
    """constant: the lower joint angle limits for a leg, offset by JOINT_LIMIT_PAD"""

    UPPER_JOINT_LIMITS = jp.array([1.047, 3.927, -0.611]) - JOINT_LIMIT_PAD
    """constant: the upper joint angle limits for a leg, offset by JOINT_LIMIT_PAD"""

    # Ограничения крутящего момента для двигателей (Нм)
    # Значения оценочные и могут требовать уточнения
    MOTOR_TORQUE_LIMIT = jp.tile(jp.array([33.5, 33.5, 40.0]), 4)
    """constant: the torque limit for the motors"""

    CACHE_PATH = epath.resource_path('brax') / 'robots/aliengo/.cache'

    @staticmethod
    def get_system(used_cached: bool = False) -> System:
        """Returns the system for the Aliengo."""

        if used_cached:
            sys = AliengoUtils._load_cached_system(approx_system=False)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/aliengo/xml/aliengo.xml'
            sys = mjcf.load(path)

        return sys

    @staticmethod
    def get_approx_system(used_cached: bool = False) -> System:
        """Returns the approximate system for the Aliengo."""

        if used_cached:
            sys = AliengoUtils._load_cached_system(approx_system=True)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/aliengo/xml/aliengo_approx.xml'
            sys = mjcf.load(path)

        return sys

    @staticmethod
    def _cache_system(approx_system: bool) -> System:
        """Cache the system for the Aliengo to avoid reloading the xml file."""
        sys = AliengoUtils.get_system()
        Path(AliengoUtils.CACHE_PATH).mkdir(parents=True, exist_ok=True)
        with open(AliengoUtils._cache_path(approx_system), 'wb') as f:
            dill.dump(sys, f)
        return sys

    @staticmethod
    def _load_cached_system(approx_system: bool) -> System:
        """Load the cached system for the Aliengo."""
        try:
            with open(AliengoUtils._cache_path(approx_system), 'rb') as f:
                sys = dill.load(f)
        except FileNotFoundError:
            sys = AliengoUtils._cache_system(approx_system)
        return sys

    @staticmethod
    def _cache_path(approx_system: bool) -> epath.Path:
        """Get the path to the cached system for the Aliengo."""
        if approx_system:
            path = AliengoUtils.CACHE_PATH / 'aliengo_approx_system.pkl'
        else:
            path = AliengoUtils.CACHE_PATH / 'aliengo_system.pkl'
        return path

    @staticmethod
    def forward_kinematics(leg: str, q: jp.ndarray) -> jp.ndarray:
        """Returns the position of the foot in the body frame centered on the
           trunk, given the joint angles; (3,)

        Arguments:
            leg (str): the name of the leg - 'FR', 'FL', 'RR', 'RL'
            q (jp.ndarray): the joint angles of a leg; (3,)
        """
        if leg not in ['FR', 'FL', 'RR', 'RL']:
            raise ValueError('leg must be one of FR, FL, RR, RL')

        side_sign = jax.lax.select(leg in ['FR', 'RR'], -1, 1)

        l1 = side_sign * AliengoUtils.THIGH_OFFSET
        l2 = -AliengoUtils.THIGH_LENGTH
        l3 = -AliengoUtils.CALF_LENGTH

        s1 = jp.sin(q[0])
        s2 = jp.sin(q[1])
        s3 = jp.sin(q[2])

        c1 = jp.cos(q[0])
        c2 = jp.cos(q[1])
        c3 = jp.cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        p0_hip = l3 * s23 + l2 * s2
        p1_hip = -l3 * s1 * c23 + l1 * c1 - l2 * c2 * s1
        p2_hip = l3 * c1 * c23 + l1 * s1 + l2 * c1 * c2

        p0 = p0_hip + jax.lax.select(leg in ['FR', 'FL'],
                                     AliengoUtils.LEG_OFFSET_X,
                                     -AliengoUtils.LEG_OFFSET_X)
        p1 = p1_hip + jax.lax.select(leg in ['FR', 'RR'],
                                     -AliengoUtils.LEG_OFFSET_Y,
                                     AliengoUtils.LEG_OFFSET_Y)
        p2 = p2_hip

        p = jp.stack([p0, p1, p2], axis=0)
        return p

    @staticmethod
    def forward_kinematics_all_legs(q: jp.ndarray) -> jp.ndarray:
        """Returns the positions of the feet in the body frame centered on the
           trunk, given the joint angles; (12,)

        Arguments:
            q (jp.ndarray): the joint angles of all legs; (12,)
        """
        p = jp.concatenate([
            AliengoUtils.forward_kinematics('FR', q[0:3]),
            AliengoUtils.forward_kinematics('FL', q[3:6]),
            AliengoUtils.forward_kinematics('RR', q[6:9]),
            AliengoUtils.forward_kinematics('RL', q[9:12]),
        ])
        return p

    @staticmethod
    def inverse_kinematics(leg: str, p: jp.ndarray) -> jp.ndarray:
        """Returns the joint angles of a leg given the position of the foot in
           the body frame centered on the trunk; (3,)

        Arguments:
            leg (str): the name of the leg - 'FR', 'FL', 'RR', 'RL'
            p (jp.ndarray): the position of the foot in the body frame; (3,)
        """
        if leg not in ['FR', 'FL', 'RR', 'RL']:
            raise ValueError('leg must be one of FR, FL, RR, RL')

        fx = jax.lax.select(leg in ['RR', 'RL'],
                            -AliengoUtils.LEG_OFFSET_X,
                            AliengoUtils.LEG_OFFSET_X)
        fy = jax.lax.select(leg in ['FR', 'RR'],
                            -AliengoUtils.LEG_OFFSET_Y,
                            AliengoUtils.LEG_OFFSET_Y)

        # Относительное положение стопы к основанию ноги
        pos_hip = jp.array([p[0] - fx, p[1] - fy, p[2]])

        # Длина проекции ноги на плоскость x-z, исходя из геометрии Aliengo
        L = jp.sqrt(pos_hip[0]**2 + pos_hip[2]**2)

        # Расчет углов с учетом геометрии Aliengo
        l_thigh = AliengoUtils.THIGH_LENGTH
        l_calf = AliengoUtils.CALF_LENGTH

        # Расчет угла в hip joint (abduction/adduction - поворот вокруг оси y)
        q1 = jp.arctan2(pos_hip[1], jp.sqrt(pos_hip[0]**2 + pos_hip[2]**2))
        if leg in ['FL', 'RL']:
            q1 = -q1

        # Расчет углов для knee и ankle
        D = (L**2 + pos_hip[1]**2 - l_thigh**2 - l_calf**2) / (2 * l_thigh * l_calf)
        # Ограничиваем D в пределах [-1, 1] для arccos
        D = jp.clip(D, -1.0, 1.0)
        
        # Угол в колене
        q3 = jp.arccos(D)
        # В робототехнике угол колена обычно отрицательный
        q3 = -q3

        # Угол в тазобедренном суставе (pitch - вокруг оси x)
        q2 = jp.arctan2(pos_hip[2], pos_hip[0]) - jp.arctan2(l_calf * jp.sin(q3), l_thigh + l_calf * jp.cos(q3))

        # Собираем вместе, меняя знаки в зависимости от ноги
        q = jp.array([q1, q2, q3])
        
        return q

    @staticmethod
    def inverse_kinematics_all_legs(p: jp.ndarray) -> jp.ndarray:
        """Returns the joint angles of all legs given the positions of the feet
           in the body frame centered on the trunk; (12,)

        Arguments:
            p (jp.ndarray): the positions of the feet in the body frame; (12,)
        """
        q = jp.concatenate([
            AliengoUtils.inverse_kinematics('FR', p[0:3]),
            AliengoUtils.inverse_kinematics('FL', p[3:6]),
            AliengoUtils.inverse_kinematics('RR', p[6:9]),
            AliengoUtils.inverse_kinematics('RL', p[9:12]),
        ])
        return q

    @staticmethod
    def jacobian(leg: str, q: jp.ndarray) -> jp.ndarray:
        """Returns the Jacobian matrix for a leg; (3, 3)

        Arguments:
            leg (str): the name of the leg - 'FR', 'FL', 'RR', 'RL'
            q (jp.ndarray): the joint angles of a leg; (3,)
        """
        if leg not in ['FR', 'FL', 'RR', 'RL']:
            raise ValueError('leg must be one of FR, FL, RR, RL')

        # Определение знака в зависимости от стороны ноги
        side_sign = jax.lax.select(leg in ['FR', 'RR'], -1, 1)

        # Параметры для вычисления якобиана
        l1 = side_sign * AliengoUtils.THIGH_OFFSET
        l2 = AliengoUtils.THIGH_LENGTH
        l3 = AliengoUtils.CALF_LENGTH

        # Вычисление синусов и косинусов углов
        s1 = jp.sin(q[0])
        c1 = jp.cos(q[0])
        s2 = jp.sin(q[1])
        c2 = jp.cos(q[1])
        s3 = jp.sin(q[2])
        c3 = jp.cos(q[2])
        s23 = jp.sin(q[1] + q[2])
        c23 = jp.cos(q[1] + q[2])

        # Вычисление элементов якобиана
        J = jp.zeros((3, 3))

        # Первый столбец
        J = J.at[0, 0].set(0)
        J = J.at[1, 0].set(-c1 * (l3 * c23 + l2 * c2) - l1 * s1)
        J = J.at[2, 0].set(-s1 * (l3 * c23 + l2 * c2) + l1 * c1)

        # Второй столбец
        J = J.at[0, 1].set(l3 * c23 + l2 * c2)
        J = J.at[1, 1].set(-s1 * (l3 * s23 + l2 * s2))
        J = J.at[2, 1].set(c1 * (l3 * s23 + l2 * s2))

        # Третий столбец
        J = J.at[0, 2].set(l3 * c23)
        J = J.at[1, 2].set(-l3 * s1 * s23)
        J = J.at[2, 2].set(l3 * c1 * s23)

        return J

    @staticmethod
    def foot_vel(leg: str, q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
        """Returns the velocity of the foot in the body frame centered on the
           trunk, given the joint angles and velocities; (3,)

        Arguments:
            leg (str): the name of the leg - 'FR', 'FL', 'RR', 'RL'
            q (jp.ndarray): the joint angles of a leg; (3,)
            qd (jp.ndarray): the joint velocities of a leg; (3,)
        """
        J = AliengoUtils.jacobian(leg, q)
        return J @ qd

    @staticmethod
    def foot_vel_all_legs(q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
        """Returns the velocities of the feet in the body frame centered on the
           trunk, given the joint angles and velocities; (12,)

        Arguments:
            q (jp.ndarray): the joint angles of all legs; (12,)
            qd (jp.ndarray): the joint velocities of all legs; (12,)
        """
        v = jp.concatenate([
            AliengoUtils.foot_vel('FR', q[0:3], qd[0:3]),
            AliengoUtils.foot_vel('FL', q[3:6], qd[3:6]),
            AliengoUtils.foot_vel('RR', q[6:9], qd[6:9]),
            AliengoUtils.foot_vel('RL', q[9:12], qd[9:12]),
        ])
        return v

    @staticmethod
    def standing_foot_positions() -> jp.ndarray:
        """Returns the standing foot positions in the body frame."""
        return AliengoUtils.STANDING_FOOT_POSITIONS 