import numpy as np
import gym
from deep_q_learning import DDPG, OrnsteinUhlenbeckActionNoise


def postprocess_action(action):
    new_action = ...
    return new_action


def preprocess_reward(reward):
    new_reward = np.clip(reward, -5., 5.)
    return new_reward


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.reset()
    env.render()

    state_space_shape = env.observation_space.shape[0]
    action_space_shape = env.action_space.shape[0]

    num_episodes = 30
    learning_rate_actor = 0.01
    learning_rate_critic = 0.02
    discount_factor = 0.99
    batch_size = 64
    memory_size = 1000

    noise = OrnsteinUhlenbeckActionNoise(action_space_shape)

    agent = DDPG(state_space_shape, action_space_shape, learning_rate_actor, learning_rate_critic,
                 discount_factor, batch_size, memory_size)

    agent.build_model()

    for episode in range(num_episodes):
        state = env.reset()
        env.render()
        done = False
        while not done:
            action = agent.get_action(state, discrete=False) + noise()
            # action = postprocess(agent.get_action(state, discrete=False) + noise())
            next_state, reward, done, _ = env.step(action)
            env.render()
            reward = preprocess_reward(reward)
            numeric_done = 1 if done == True else 0
            agent.update_memory(state, action, reward, next_state, numeric_done)
            state = next_state
        agent.train()
        if episode % 5 == 0:
            agent.update_target_model()
        if episode % 50 == 0:
            agent.save('pendulum', episode)
