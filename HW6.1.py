########################################################################
# Mantz Wyrick
# Machine Learning Homework #6.1
#
# Reinforcement learning: Q-learning "frozen lake"
#
########################################################################

"""
A frozenlake-v0 is a 4x4 grid world which looks as follows:
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
Additionally, there is a little uncertainity in the agent movement.
Q Learning - A simple q Learning algorithm is employed for the task.
The q values are stored in a table and these are updated in each iteration
to converge to their optimum values.
"""

import gym
import numpy as np
import random
import math


def main():
    env = gym.make("FrozenLake-v0")

    num_episodes = 100000
    max_steps = 500
    gamma = 0.99
    learning_rate = 0.1
    discount_rate = 0.99
    epsilon = 1.5
    max_epsilon = 1.5
    min_epsilon = 0.01
    decay_rate = 0.01

    # initialize the Q table
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    Q = np.zeros([state_space_size, action_space_size])
    rewards = []

    for i in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        for j in range(max_steps):
            # pick a random number to decide between explore and exploit
            if random.uniform(0, 1.5) < epsilon:
                action = env.action_space.sample()  # explore
            else:
                action = np.argmax(Q[state])  # exploit
            # if exploit, use Q value to pick action
            # if explore, pick randomly with env.action_space.sample()

            # update state, reward, done, and info based on taking action
            next_state, reward, done, info = env.step(action)

            # update total_reward
            total_reward = reward

            # update Q values
            old_value = Q[state, action]
            next_max = np.max(Q[next_state])

            new_value = (1 - learning_rate) * old_value + learning_rate * (
                reward + discount_rate * next_max
            )
            Q[state, action] = new_value

            # update state
            state = next_state

            # if done, break
            if done:
                break

            if i == (num_episodes - 1):  # see a visualization of the agent
                env.render()
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i)
        rewards.append(total_reward)
    print(np.around(Q, 6))
    print("score:", np.mean(rewards))


main()
