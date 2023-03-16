import pandas as pd
import os
from GAMEE import ChopperScape, build_model, build_conv2d_model, build_keras_model
from av5.deep_q_learning import DDQN, DQN
from tqdm import tqdm
import time

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    #
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    env = ChopperScape()

    # env.render()
    env.reset()
    env.render()

    state_space_shape = env.observation_space.shape
    num_actions = env.action_space.n
    max_steps = 1000
    num_episodes = 3000
    learning_rate = 0.01
    batch_size = 32
    memory_size = 1024
    epsilon = 0.1
    decay = 0.000
    discount_factor = 0.9

    model = build_model(state_space_shape, num_actions, learning_rate)
    target_model = build_model(state_space_shape, num_actions, learning_rate)
    # k_model = build_keras_model(state_space_shape, num_actions)
    # c2d_model_target = build_conv2d_model(state_space_shape, num_actions, learning_rate)
    # c2d_model = build_conv2d_model(state_space_shape, num_actions, learning_rate)
    agent = DDQN(state_space_shape, num_actions, model, target_model, learning_rate, discount_factor, batch_size,
                 memory_size)

    # episode_summary = []
    # for episode in tqdm(range(1, num_episodes + 1)):
    #     state = env.reset()
    #     done = False
    #     steps = 0
    #     sum = 0
    #     while not done and steps < max_steps:
    #         t = time.time()
    #         # print(steps)
    #         action = agent.get_action(state, epsilon)
    #         new_state, reward, done, _ = env.step(action)
    #         # env.render()
    #         agent.update_memory(state, action, reward, new_state, done)
    #         state = new_state
    #         steps += 1
    #         print("time_per_step: ", time.time() - t)
    #     agent.train()
    #     agent.update_target_model()
    #     epsilon -= decay
    #     # print(epsilon)
    #     if steps == max_steps:
    #         episode_summary.append([episode, env.ep_return, steps, env.fuel_left, "MAX_STEPS"])
    #     elif env.ep_return >= 1200:
    #         episode_summary.append([episode, env.ep_return, steps, env.fuel_left, "WON"])
    #     elif env.fuel_left == 0:
    #         episode_summary.append([episode, env.ep_return, steps, env.fuel_left, "NO_FUEL"])
    #     else:
    #         episode_summary.append([episode, env.ep_return, steps, env.fuel_left, "DIED"])
    #
    #     if episode % 200 == 0:
    #         agent.save("saved", episode)
    #
    # df = pd.DataFrame(episode_summary, columns=['Episode', 'Score', 'Steps', 'Remaining_Fuel', 'Cause'])
    # df.to_csv("Training_DDQN_3000EP", index=False)
    agent.load("saved", 8000)
    done = False
    state = env.reset()
    env.render()
    total_reward = 0
    while not done:
        action = agent.get_action(state, epsilon)
        # action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        print(action)
        # Render the game
        # print(obs)
        env.render()

        if done == True:
            print(total_reward)
            break
    # print(df)
    env.close()
