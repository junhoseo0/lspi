import sys
import os.path
import random
from itertools import product
from typing import Callable

import gymnasium as gym
import numpy as np
from scipy.spatial.distance import cdist
from gymnasium.wrappers.normalize import NormalizeObservation

sys.modules["gym"] = gym
from stable_baselines3 import DQN


def gaussian_radials(centers: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """
    Parameters:
    ----------
    centers: numpy.ndarray
        a (n, obs_shape)-shaped array.
    observations: numpy.ndarray
        a (m, obs_shape)-shaped array.

    Returns
    -------
    radials: numpy.ndarray
        a (n, m)-shaped array
    """

    # dists is (n, m)-shaped array.
    dists = cdist(observations, centers)
    radials = np.exp(-dists)
    return radials


def rollout(
    env: gym.Env, buffer_size: int, action_selector: Callable[[np.ndarray], int]
) -> dict[str, np.ndarray]:
    # Assume one-dimensional Box space
    obs_size = env.observation_space.shape[0]
    buffer = {
        "observations": np.zeros((buffer_size, obs_size)),
        "actions": np.zeros((buffer_size,), dtype=np.int32),
        "rewards": np.zeros((buffer_size,)),
    }

    t = 0
    while t < buffer_size:
        done = False
        obs, _ = env.reset()
        while (not done) and t < buffer_size:
            action = action_selector(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Update the buffer
            buffer["observations"][t] = np.array(obs)
            buffer["actions"][t] = np.array(action)
            buffer["rewards"][t] = np.array(reward)

            done = terminated or truncated
            obs = next_obs
            t += 1

    return buffer


def mix_buffers(
    buffer_size: int,
    buffer1: dict[str, np.ndarray],
    buffer2: dict[str, np.ndarray],
    tau: float = 0.5,
):
    orig = int(buffer_size * tau)
    random_idxs = random.sample(range(buffer_size), orig)

    buffer = {}
    for key in buffer1:
        mask = np.zeros_like(buffer1[key])
        mask[random_idxs] = 1.0

        buffer[key] = buffer1[key] * mask + buffer2[key] * (1 - mask)
    return buffer


def main():
    # Define arguments
    buffer_size = 10000
    k = 3
    gamma = 0.99
    stop_criterion = 1.0

    random_tau = 1.0
    env_name = "MountainCar-v0"

    # Intialize the environment
    env = gym.make(env_name)
    env = NormalizeObservation(env)

    # Populate environmental informations
    num_actions = env.action_space.n  # Assume Discrete space

    # Populate centers using environmental infromations
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    centers = np.linspace(obs_low, obs_high, k).T.tolist()
    centers = np.array(list(product(*centers)))

    # Learn or load DQN policy
    if os.path.exists(f"{env_name}_dqn_model.zip"):
        model = DQN.load(f"{env_name}_dqn_model")
    else:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            train_freq=16,
            gradient_steps=8,
            gamma=gamma,
            exploration_fraction=0.2,
            exploration_final_eps=0.07,
            target_update_interval=600,
            learning_starts=1000,
            buffer_size=10000,
            batch_size=128,
            learning_rate=4e-3,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
        model.learn(total_timesteps=120000, log_interval=10)
        model.save(f"{env_name}_dqn_model")

    # Generate random and DQN trajectories and mix them
    buffer_random = rollout(env, buffer_size, lambda _: env.action_space.sample())
    buffer_dqn = rollout(
        env, buffer_size, lambda obs: model.predict(obs, deterministic=True)[0]
    )
    buffer = mix_buffers(buffer_size, buffer_random, buffer_dqn, random_tau)

    # Delete the original buffers after mixing to save a memory.
    del buffer_random
    del buffer_dqn

    # Pre-compute all radials
    radials = gaussian_radials(
        centers, buffer["observations"]
    )  # (buffer_size, feature_size)
    feature_size = radials.shape[1]

    radials_ext = np.tile(
        radials, reps=[1, num_actions]
    )  # (buffer_size, num_actions * feature_size)

    # Pre-compute features for all observed observations and actions
    r = np.arange(num_actions * feature_size)
    action_idxs = np.arange(num_actions)
    feature_mask = (action_idxs[:, None] * feature_size <= r) & (
        (action_idxs[:, None] + 1) * feature_size > r
    )

    features = np.zeros((buffer_size, num_actions, num_actions * feature_size))
    features[:, feature_mask] = radials_ext  # (buffer_size, num_actions, feature_size)

    features_observed = np.take_along_axis(
        features, buffer["actions"][:, None, None], axis=1
    )

    # Pre-compute A and b for all transitions
    # Row-wise outer product; (buffer_size, num_actions, feature_size, feature_size)
    matA_all = np.matmul(
        features_observed[:-1][:, :, :, None],
        (features_observed[:-1] - gamma * features[1:])[:, :, None, :],
    )

    # NOTE: numpy.take_along_axis is slow operation, though necessary.
    b = (buffer["rewards"][:, None, None] * features_observed).squeeze()
    b = np.sum(b, axis=0)  # (feature_size,)

    # LSPI
    epoch = 0
    np_rng = np.random.default_rng()
    weight = np_rng.normal(size=(num_actions, feature_size))
    new_weight = np.array(weight)
    while True:
        # Choose greedy actions
        values = np.dot(weight, radials[1:].T)
        actions = np.argmax(values, axis=0)

        # Update the weight
        # NOTE: numpy.take_along_axis is slow operation, though necessary.
        matA = np.take_along_axis(
            matA_all, actions[:, None, None, None], axis=1
        ).squeeze()
        matA = np.sum(matA, axis=0)
        new_weight = np.linalg.solve(matA, b).reshape(num_actions, feature_size)
        distance = np.linalg.norm(weight - new_weight)
        weight = new_weight

        print(f"{epoch}: {distance}")
        epoch += 1

        if distance < stop_criterion:
            break

    # Run the resulting policy
    done = False
    action_list = []
    ep_return = 0.0
    obs, _ = env.reset()
    while not done:
        feature = gaussian_radials(centers, obs[None, :]).squeeze()
        values = np.dot(weight, feature)
        action = np.argmax(values, axis=-1)
        action_list.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += reward

        done = terminated or truncated

    print(ep_return)
    print(action_list)

    # Clean up
    env.close()


if __name__ == "__main__":
    main()
