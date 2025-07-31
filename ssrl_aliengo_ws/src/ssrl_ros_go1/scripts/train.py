#!/usr/bin/env python
from brax.envs.go1_go_fast import Go1GoFast
from brax.robots.go1 import networks as go1_networks
from brax.training.acme import running_statistics
from brax.training.agents.ssrl import train as ssrl
from brax.training.agents.ssrl import base as ssrl_base
from brax.envs.base import RlwamEnv
from brax.training.types import Transition, PRNGKey
from ssrl_ros_go1.bag import read_bag
from ssrl_ros_go1.plot_rollout import plot_rollout
from ssrl_ros_go1 import env_dict

import time
import os
import re
from typing import Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
import functools as ft
from pathlib import Path
import dill
import jax
import flax
import optax
from jax import numpy as jp
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # Добавлен TensorBoard
import numpy as np

data_path = (Path(os.path.abspath(__file__)).parent.parent / 'data')


@flax.struct.dataclass
class ssrlData:
    ssrl_state: ssrl_base.TrainingState
    env_buffer_state: ssrl_base.ReplayBufferState
    model_buffer_state: ssrl_base.ReplayBufferState
    key: PRNGKey


@flax.struct.dataclass
class StepsState:
    steps: int


@hydra.main(config_path="configs", config_name="go1")
def train(cfg: DictConfig):
    from jax import config
    config.update("jax_enable_x64", True)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)

    # Инициализация TensorBoard
    log_dir = data_path / cfg.run_name / "tensorboard"
    writer = SummaryWriter(log_dir=str(log_dir))

    # Инициализация обучения
    ms, env_unwrapped, epoch, rollout_path, steps, warm_start = init_training(cfg, writer)

    # Обновление горизонта модели
    ms = ssrl.update_model_horizon(ms, epoch)

    # Загрузка данных rollout
    ms, rollout_metrics = load_rollout(ms, cfg, env_unwrapped, epoch, rollout_path, writer, steps)
    steps += rollout_metrics['rollout_steps']
    print(rollout_metrics)

    # Обучение
    print("Начало обучения со следующими параметрами:")
    ssrl_params = OmegaConf.to_container(cfg.ssrl, resolve=True, throw_on_missing=True)
    print(ssrl_params)
    print("Обучение...")

    # Обучение модели
    model_train_time = 0
    other_time = 0

    # Обучение модели
    new_key, model_key = jax.random.split(ms.local_key)
    ms = ms.replace(local_key=new_key)
    start_time = time.time()
    c = ms.constants.replace(model_training_max_epochs=1) if warm_start else ms.constants
    training_state, model_metrics = ssrl.train_model(
        ms.training_state, ms.env_buffer_state, ms.env_buffer,
        c, model_key)
    ms = ms.replace(training_state=training_state)
    model_train_time += time.time() - start_time
    if ms.constants.clear_model_buffer_after_model_train:
        rb = ms.sac_state.replay_buffer
        bs = rb.init(ms.sac_state.buffer_state.key)
        ms = ms.replace(sac_state=ms.sac_state.replace(buffer_state=bs))

    # Обновление политики
    start_time = time.time()
    update_key, new_local_key = jax.random.split(ms.local_key)
    ms = ms.replace(local_key=new_local_key)
    (training_state, sac_training_state, env_buffer_state, sac_buffer_state,
     sac_metrics) = ssrl.policy_update(
        ms.training_state, ms.sac_state.training_state, ms.constants,
        ms.sac_state.constants, ms.env_buffer_state, ms.env_buffer,
        ms.sac_state.buffer_state, ms.sac_state.replay_buffer, ms.model_env,
        update_key, ms.model_horizon,
        ms.hallucination_updates_per_training_step)
    other_time += time.time() - start_time
    ms = ms.replace(
        training_state=training_state,
        sac_state=ms.sac_state.replace(training_state=sac_training_state,
                                       buffer_state=sac_buffer_state),
        env_buffer_state=env_buffer_state)
    sac_metrics = jax.tree_util.tree_map(jp.mean, sac_metrics)

    metrics = {
        'steps': steps,
        'epoch': epoch,
        'training/model_train_time': model_train_time,
        'training/other_time': other_time,
        'training/model_horizon': ms.model_horizon,
        'training/hallucination_updates_per_training_step': (
                    ms.hallucination_updates_per_training_step),
        'training/env_buffer_size': ms.env_buffer.size(ms.env_buffer_state),
        'training/reset_critic': 1 if cfg.reset_critic else 0,
        **{f'model/{name}': value for name, value in model_metrics.items()},
        **{f'sac/{name}': value for name, value in sac_metrics.items()},
        **{f'rollout/{name}': value for name, value in rollout_metrics.items()},
    }
    print(metrics)

    # Логирование метрик в TensorBoard
    for key, value in metrics.items():
        # Преобразуем JAX-массивы к numpy, а numpy к float
        if hasattr(value, "item"):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = float(value)
        writer.add_scalar(key, value, global_step=steps)

    # Сохранение состояний
    sac_ts_path = (rollout_path / 'sac_ts.pkl')
    with open(sac_ts_path, 'wb') as f:
        dill.dump(sac_training_state, f)

    ssrl_data_path = (rollout_path / 'ssrl_data.pkl')
    if not ms.constants.clear_model_buffer_after_model_train:
        model_buffer_state = sac_buffer_state
    else:
        model_buffer_state = None
    ssrl_data = ssrlData(ssrl_state=training_state,
                             env_buffer_state=env_buffer_state,
                             model_buffer_state=model_buffer_state,
                             key=ms.local_key)
    with open(ssrl_data_path, 'wb') as f:
        dill.dump(ssrl_data, f)

    # Сохранение состояния шагов
    steps_path = (rollout_path / 'steps_state.pkl')
    steps_state = StepsState(steps=steps)
    with open(steps_path, 'wb') as f:
        dill.dump(steps_state, f)

    # Закрытие TensorBoard
    writer.close()


def init_training(cfg: DictConfig, writer: SummaryWriter) -> Tuple[ssrl_base.MbpoState, RlwamEnv, int, Path, int, bool]:
    rollout_folders = [folder for folder in os.listdir(data_path / cfg.run_name) if folder.isdigit()]
    if not rollout_folders:
        rollout_num = 0
    else:
        rollout_num = max(int(folder) for folder in rollout_folders)
    print(f'Загрузка rollout {rollout_num}')
    rollout_path = (data_path / cfg.run_name / f"{rollout_num:02d}")

    # Загрузка состояния шагов
    steps = 0
    if rollout_num > 0:
        prev_rollout_path = (data_path / cfg.run_name / f"{rollout_num-1:02d}")
        steps_path = prev_rollout_path / "steps_state.pkl"
        if os.path.exists(steps_path):
            with open(steps_path, 'rb') as f:
                steps_state = dill.load(f)
                steps = steps_state.steps

    # Загрузка состояния SAC
    sac_ts_path = (data_path / cfg.run_name / f"{rollout_num-1:02d}" / "sac_ts.pkl")
    if os.path.exists(sac_ts_path):
        with open(sac_ts_path, 'rb') as f:
            sac_ts = dill.load(f)
    else:
        sac_ts = None

    # Загрузка состояния SSRL
    ssrl_data_path = (data_path / cfg.run_name / f"{rollout_num-1:02d}" / "ssrl_data.pkl")
    if os.path.exists(ssrl_data_path):
        with open(ssrl_data_path, 'rb') as f:
            ssrl_data = dill.load(f)
    else:
        ssrl_data = None

    # Проверка предобученной модели
    warm_start_ssrl_data = None
    warm_start_ssrl_data_path = (data_path / cfg.run_name / "00" / "ssrl_data.pkl")
    if rollout_num == 0 and os.path.exists(warm_start_ssrl_data_path):
        with open(warm_start_ssrl_data_path, 'rb') as f:
            warm_start_ssrl_data = dill.load(f)

    env_kwargs = cfg.env_ssrl
    env_fn = ft.partial(env_dict[cfg.env], backend='generalized')
    env_fn = add_kwargs_to_fn(env_fn, **env_kwargs)
    env = env_fn()

    dynamics_fn = env.make_ssrl_dynamics_fn(cfg.ssrl_dynamics_fn)
    (sac_network_factory,
        model_network_factory) = go1_networks.ssrl_network_factories(cfg)

    if cfg.reset_critic and sac_ts is not None:
        q_optimizer = optax.adam(learning_rate=cfg.ssrl.sac_learning_rate)
        normalize_fn = lambda x, y: x  # noqa: E731
        if cfg.common.normalize_observations:
            normalize_fn = running_statistics.normalize
        sac_network = sac_network_factory(
            observation_size=env.observation_size*cfg.common.obs_history_length,
            action_size=env.action_size,
            preprocess_observations_fn=normalize_fn
        )
        key_q = jax.random.PRNGKey(cfg.ssrl.seed)
        q_params = sac_network.q_network.init(key_q)
        q_optimizer_state = q_optimizer.init(q_params)
        sac_ts = sac_ts.replace(
                q_params=q_params, q_optimizer_state=q_optimizer_state)

    if cfg.reset_actor and sac_ts is not None:
        policy_optimizer = optax.adam(learning_rate=cfg.ssrl.sac_learning_rate)
        sac_network = sac_network_factory(
            observation_size=env.observation_size*cfg.common.obs_history_length,
            action_size=env.action_size,
            preprocess_observations_fn=normalize_fn
        )
        key_policy = jax.random.PRNGKey(cfg.ssrl.seed)
        policy_params = sac_network.policy_network.init(key_policy)
        policy_optimizer_state = policy_optimizer.init(policy_params)
        sac_ts = sac_ts.replace(
                policy_params=policy_params,
                policy_optimizer_state=policy_optimizer_state)

    output_dim = (env.observation_size if cfg.ssrl_dynamics_fn == 'mbpo'
                  else env.sys.qd_size())

    if cfg.reset_model and ssrl_data is not None:
        model_optimizer = optax.adam(cfg.ssrl.model_learning_rate)
        obs_stack_size = env.observation_size*cfg.common.obs_history_length
        model_network = model_network_factory(env.observation_size, output_dim)
        dummy_X = jp.zeros((model_network.ensemble_size,
                            obs_stack_size + env.action_size))
        key_model = jax.random.PRNGKey(cfg.ssrl.seed)
        model_params = model_network.init(key_model, dummy_X)
        model_optimizer_state = model_optimizer.init(model_params['params'])
        ssrl_data = ssrl_data.replace(
            ssrl_state=ssrl_data.ssrl_state.replace(
                model_params=model_params,
                model_optimizer_state=model_optimizer_state))

    if cfg.ssrl.sac_fixed_alpha == 'None':
        cfg.ssrl.sac_fixed_alpha = None
    init_train_fn = ft.partial(ssrl.initizalize_training)
    init_train_fn = add_kwargs_to_fn(init_train_fn, **cfg.ssrl)
    model_horizon_fn = ssrl.make_linear_threshold_fn(
        cfg.ssrl_linear_threshold_fn.start_epoch,
        cfg.ssrl_linear_threshold_fn.end_epoch,
        cfg.ssrl_linear_threshold_fn.start_model_horizon,
        cfg.ssrl_linear_threshold_fn.end_model_horizon)
    hupts_fn = ssrl.make_linear_threshold_fn(
        cfg.ssrl_hupts_fn.start_epoch,
        cfg.ssrl_hupts_fn.end_epoch,
        cfg.ssrl_hupts_fn.start_hupts,
        cfg.ssrl_hupts_fn.end_hupts)
    ssrl_state = init_train_fn(
        environment=env,
        low_level_control_fn=env.low_level_control,
        dynamics_fn=dynamics_fn,
        reward_fn=env.ssrl_reward_fn,
        model_output_dim=output_dim,
        model_horizon_fn=model_horizon_fn,
        hallucination_updates_per_training_step=hupts_fn,
        sac_training_state=sac_ts,
        model_network_factory=model_network_factory,
        sac_network_factory=sac_network_factory,
        num_timesteps=0,
        progress_fn=lambda *args: None,
        policy_params_fn=lambda *args: None,
        eval_env=env
    )

    if ssrl_data is not None:
        ssrl_state = ssrl_state.replace(
            training_state=ssrl_data.ssrl_state,
            env_buffer_state=ssrl_data.env_buffer_state)
        if hasattr(ssrl_data, 'key'):
            ssrl_state = ssrl_state.replace(local_key=ssrl_data.key)
        else:
            ssrl_state = ssrl_state.replace(
                local_key=jax.random.PRNGKey(cfg.ssrl.seed))
        if not cfg.ssrl.clear_model_buffer_after_model_train:
            ssrl_state = ssrl_state.replace(
                sac_state=ssrl_state.sac_state.replace(
                    buffer_state=ssrl_data.model_buffer_state))

    if warm_start_ssrl_data is not None:
        ssrl_state = ssrl_state.replace(
            training_state=warm_start_ssrl_data.ssrl_state)
    warm_start = True if warm_start_ssrl_data is not None else False

    return ssrl_state, env, rollout_num, rollout_path, steps, warm_start


def load_rollout(ms: ssrl_base.MbpoState, cfg: DictConfig,
                 env: Go1GoFast, rollout_num: int, rollout_path: Path,
                 writer: SummaryWriter, current_steps: int):
    obs_size = env.observation_size
    hist_len = cfg.common.obs_history_length
    act_repeat = cfg.common.action_repeat
    act_size = env.action_size
    q_size = env.sys.act_size()
    u_size = env.controls_size
    is_straight_task = cfg.env == 'Go1GoFast'

    bag_paths = []
    pattern = re.compile(r'subrollout_\d+\.bag')
    for file in os.listdir(rollout_path):
        if pattern.match(file):
            bag_paths.append(rollout_path / file)
    bag_paths = sorted(bag_paths)

    # Для метрик rollout
    ep_lens = jp.array([])
    forward_vels = jp.array([])
    first_avg_forward_vel = -jp.inf
    side_vels = jp.array([])
    if is_straight_task:
        turn_rates = jp.array([])
    else:
        delta_radii = jp.array([])
        delta_yaw = jp.array([])
    all_reward_components = []

    # Для построения графиков
    all_ts_plot = None
    all_obses_plot = jp.zeros((0, obs_size))
    all_q_deses_plot = jp.zeros((0, q_size))
    all_actions_plot = jp.zeros((0, act_size))
    all_theo_torques_plot = jp.zeros((0, q_size))
    all_theo_energy_plot = jp.zeros((0, 1))

    # Для обучения
    all_norm_obses_stack = jp.zeros((0, obs_size*hist_len))
    all_actions = jp.zeros((0, act_size))
    all_us = jp.zeros((0, u_size))
    all_rewards = jp.zeros((0,))
    subrollout_rewards = jp.zeros((0,))
    all_dones = jp.zeros((0,))

    for i, bag_path in enumerate(bag_paths):
        # Загрузка данных из rosbag
        print('Загрузка данных')
        ts, obses, qs, qds, q_deses, qd_deses, Kps, Kds, actions = read_bag(
            bag_path,
            q_idxs=env._q_idxs,
            qd_idxs=env._qd_idxs
        )
        norm_obses = [env._normalize_obs(obs) for obs in obses]

        # Формирование данных для графиков
        if all_ts_plot is None:
            all_ts_plot = jp.array(ts)
        else:
            all_ts_plot = jp.concatenate([all_ts_plot,
                                        jp.array(ts) + 2*all_ts_plot[-1] - all_ts_plot[-2]],
                                    axis=0)
        all_obses_plot = jp.concatenate([all_obses_plot, jp.array(obses)], axis=0)
        all_q_deses_plot = jp.concatenate([all_q_deses_plot, jp.array(q_deses)], axis=0)
        all_actions_plot = jp.concatenate([all_actions_plot, jp.array(actions)], axis=0)
        theo_torques = (jp.array(Kps) * (jp.array(q_deses) - jp.array(qs))
                        + jp.array(Kds) * (jp.array(qd_deses) - jp.array(qds)))
        all_theo_torques_plot = jp.concatenate([all_theo_torques_plot, theo_torques], axis=0)
        theo_energy = jp.expand_dims(jp.sum(jp.abs(theo_torques * jp.array(qds)), axis=-1), axis=-1)
        all_theo_energy_plot = jp.concatenate([all_theo_energy_plot, theo_energy], axis=0)

        # Добавление данных для метрик rollout
        obses = jp.array(obses)
        ep_lens = jp.concatenate([ep_lens, jp.array([obses.shape[0]])])
        forward_vels = jp.concatenate([forward_vels, obses[:, env._forward_vel_idx]])
        if i == 0:
            first_avg_forward_vel = jp.mean(obses[:, env._forward_vel_idx])
        side_vels = jp.concatenate([side_vels, obses[:, env._y_vel_idx]])
        if is_straight_task:
            turn_rates = jp.concatenate([turn_rates, obses[:, env._turn_rate_idx]])
        else:
            delta_radii = jp.concatenate([delta_radii, obses[:, env._delta_radius_idx]])
            delta_yaw = jp.concatenate([delta_yaw, obses[:, env._delta_yaw_idx]])

        # Создание стеков наблюдений
        print('Создание стеков наблюдений')
        norm_obses_stack = []
        norm_obses_stack.append(jp.zeros((obs_size*hist_len,)))
        norm_obses_stack[0] = norm_obses_stack[0].at[:obs_size].set(norm_obses[0])
        for (i, ob) in enumerate(norm_obses):
            if i == 0:
                continue
            norm_obses_stack.append(jp.concatenate(
                [ob, norm_obses_stack[i-1][:obs_size*(hist_len-1)]],
                axis=-1
            ))
        norm_obses_stack = jp.array(norm_obses_stack)

        # Масштабирование действий
        us = env.scale_action(jp.array(actions))

        print('Вычисление наград')
        rewards = []
        rew_fn = jax.jit(env.compute_reward)
        for i in range(len(norm_obses_stack)):
            if i == 0:
                prev_norm_obs = norm_obses[0]
            else:
                prev_norm_obs = norm_obses[i-1]
            reward, reward_components = rew_fn(norm_obses[i], prev_norm_obs,
                                               us[i], actions[i])
            rewards.append(reward)
            all_reward_components.append(reward_components)

        # Обрезка неполных стеков
        trunc = ((hist_len // act_repeat)) * act_repeat
        norm_obses_stack = norm_obses_stack[trunc:]
        actions = actions[trunc:]
        us = us[trunc:]
        rewards = rewards[trunc:]

        # Обрезка для кратности action_repeat
        steps = len(norm_obses_stack) // act_repeat * act_repeat
        norm_obses_stack = jp.array(norm_obses_stack[:steps])
        actions = jp.array(actions[:steps])
        us = jp.array(us[:steps])
        rewards = jp.array(rewards[:steps])

        # Создание флагов завершения
        dones = jp.zeros_like(rewards)
        dones = dones.at[-1].set(1)

        # Объединение данных
        all_norm_obses_stack = jp.concatenate([all_norm_obses_stack, norm_obses_stack], axis=0)
        all_actions = jp.concatenate([all_actions, actions], axis=0)
        all_us = jp.concatenate([all_us, us], axis=0)
        all_rewards = jp.concatenate([all_rewards, rewards], axis=0)
        subrollout_rewards = jp.concatenate(
            [subrollout_rewards,
             jp.expand_dims(jp.sum(rewards), axis=0)], axis=0)
        all_dones = jp.concatenate([all_dones, dones], axis=0)

    # Следующие наблюдения - сдвинутые стеки
    all_next_obs_stack = jp.roll(all_norm_obses_stack, shift=-1, axis=0)

    # Построение переходов
    print('Построение переходов')
    zeros = jp.zeros_like(all_rewards)
    truncation = zeros
    info = {'truncation': truncation}
    extra_fields = ('truncation',)
    state_extras = {x: info[x] for x in extra_fields}
    extras = {'state_extras': state_extras}
    transitions = Transition(
        observation=all_norm_obses_stack,
        action=all_actions,
        reward=all_rewards,
        discount=1-all_dones,
        next_observation=all_next_obs_stack,
        extras=extras)

    # Добавление переходов в буфер
    env_buffer_state = ms.env_buffer.insert(ms.env_buffer_state, transitions)
    model_buffer_state = ms.sac_state.replay_buffer.insert(
        ms.sac_state.buffer_state, transitions)
    ms = ms.replace(
        env_buffer_state=env_buffer_state,
        sac_state=ms.sac_state.replace(buffer_state=model_buffer_state))

    # Расчет метрик rollout
    rollout_steps = all_rewards.shape[0]
    total_rewards = jp.sum(all_rewards)
    avg_reward_per_step = total_rewards / rollout_steps
    print(f'Шаги rollout: {rollout_steps}')
    print(f'Общая награда / шаги rollout: {total_rewards / rollout_steps}')

    all_reward_components = jax.tree_util.tree_map(
        lambda *x: jp.stack(x),
        *all_reward_components
    )
    all_reward_components = jax.tree_util.tree_map(
        lambda x: jp.mean(x),
        all_reward_components
    )

    rollout_metrics = {'stats/avg_episode_length': jp.mean(ep_lens),
                       'stats/max_episode_length': jp.max(ep_lens),
                       'stats/avg_forward_vel': jp.mean(forward_vels),
                       'stats/first_avg_forward_vel': first_avg_forward_vel,
                       'stats/avg_side_vel': jp.mean(side_vels),
                       'rollout_steps': rollout_steps,
                       'stats/avg_reward_per_step': avg_reward_per_step,
                       'reward': jp.max(subrollout_rewards),
                       'total_reward': total_rewards,
                       **{f'rew/{name}': value for name, value in all_reward_components.items()}}
    if is_straight_task:
        rollout_metrics['stats/avg_turn_rate'] = jp.mean(turn_rates)
    else:
        rollout_metrics['stats/avg_delta_radius'] = jp.mean(delta_radii)
        rollout_metrics['stats/avg_delta_yaw'] = jp.mean(delta_yaw)

    # Генерация графиков для TensorBoard
    if cfg.tensorboard.log_images:
        fig = plot_rollout(cfg, env, all_ts_plot,
                           obses=all_obses_plot, qs=None, qds=None, q_deses=all_q_deses_plot,
                           qd_deses=None, Kps=None, Kds=None, actions=all_actions_plot,
                           theo_torques=all_theo_torques_plot,
                           theo_energy=all_theo_energy_plot,
                           rollout_num=rollout_num, return_fig=True)
        writer.add_figure('rollout/plot', fig, global_step=current_steps)
        plt.close(fig)

    return ms, rollout_metrics


def add_kwargs_to_fn(partial_fn, **kwargs):
    """Добавляет kwargs к частичной функции"""
    for param in kwargs:
        partial_fn.keywords[param] = kwargs[param]
    return partial_fn


if __name__ == '__main__':
    train()