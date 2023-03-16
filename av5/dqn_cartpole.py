import gym
from keras.layers import Dense
from keras.losses import MeanSquaredError
from keras.models import Sequential
from keras.optimizers import adam_v2

from deep_q_learning import DQN


def build_model(state_space_shape, num_actions, learning_rate):
    model = Sequential()

    model.add(Dense(32, activation='relu', input_dim=state_space_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))

    model.compile(adam_v2.Adam(lr=learning_rate), loss=MeanSquaredError())

    return model


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.reset()
    env.render()

    state_space_shape = env.observation_space.shape[0]
    num_actions = env.action_space.n

    num_episodes = 50
    learning_rate = 0.01
    discount_factor = 0.99
    batch_size = 128
    memory_size = 1024
    epsilon = 0.1

    model = build_model(state_space_shape, num_actions, learning_rate)
    target_model = build_model(state_space_shape, num_actions, learning_rate)

    agent = DQN(state_space_shape, num_actions, model, target_model,
                learning_rate, discount_factor, batch_size, memory_size)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state, epsilon)
            new_state, reward, done, _ = env.step(action)
            agent.update_memory(state, action, reward, new_state, done)
            state = new_state
        agent.train()
        agent.update_target_model()

    # agent.load('cartpole', 3000)

    done = False
    state = env.reset()
    env.render()
    while not done:
        action = agent.get_action(state, 0)
        state, reward, done, _ = env.step(action)
        env.render()

    print()
