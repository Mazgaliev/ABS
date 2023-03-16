import random

import cv2
import numpy as np
from gym import Env, spaces
from keras.layers import Dense, Conv2D,MaxPooling2D
from keras.layers import Flatten
from keras.layers import Input
from keras.losses import MeanSquaredError
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import adam_v2

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


import random

import cv2
import numpy as np
from gym import Env, spaces
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name

    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)

    def get_position(self):
        return (self.x, self.y)

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)


class Chopper(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Chopper, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("./graphics/chopper.png") / 255.0
        self.icon_w = 64
        self.icon_h = 64
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class Bird(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Bird, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("./graphics/bird.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class Fuel(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Fuel, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("./graphics/fuel.png") / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class ChopperScape(Env):

    def __init__(self):
        super(ChopperScape, self).__init__()

        # 2D observ space
        self.total_reward = 0
        self.observation_shape = (400, 600, 3)

        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.ones(self.observation_shape),
                                            dtype=np.float16)

        # Actiosn ranging from 0-6
        self.action_space = spaces.Discrete(4)

        # Create a canvas to render the environment images upon
        self.canvas = np.ones(self.observation_shape) * 1

        # elements from environment
        self.elements = []

        self.max_fuel = 1000

        self.y_min = int(self.observation_shape[0] * 0.1)
        self.x_min = 0

        self.y_max = int(self.observation_shape[0] * 0.9)
        self.x_max = int(self.observation_shape[1])

    def draw_elements_on_canvas(self):
        # Init the canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw the heliopter on canvas
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x, y = elem.x, elem.y
            self.canvas[y: y + elem_shape[1], x:x + elem_shape[0]] = elem.icon

        text = 'Fuel Left: {} | Rewards: {}'.format(self.fuel_left, self.ep_return)

        # Put the info on canvas
        self.canvas = cv2.putText(self.canvas, text, (10, 20), font,
                                  0.8, (0, 0, 0), 1, cv2.LINE_AA)

    def has_collided(self, elem1, elem2):
        x_col = False
        y_col = False

        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()

        if 2 * abs(elem1_x - elem2_x) <= (elem1.icon_w + elem2.icon_w):
            x_col = True

        if 2 * abs(elem1_y - elem2_y) <= (elem1.icon_h + elem2.icon_h):
            y_col = True

        if x_col and y_col:
            return True

        return False

    def step(self, action):
        # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        # Decrease the fuel counter
        self.fuel_left -= 1

        # Reward for executing a step.
        reward = 1

        # apply the action to the chopper
        if action == 0:
            self.chopper.move(0, 15)
        elif action == 1:
            self.chopper.move(0, -15)
        elif action == 2:
            self.chopper.move(15, 0)
        elif action == 3:
            self.chopper.move(-10, 0)
        elif action == 4:
            self.chopper.move(0, 0)

        # Spawn a bird at the right edge with prob 0.01
        if random.random() < 0.01:
            # Spawn a bird
            spawned_bird = Bird("bird".format(self.bird_count), self.x_max, self.x_min, self.y_max, self.y_min)
            self.bird_count += 1

            # Compute the x,y co-ordinates of the position from where the bird has to be spawned
            # Horizontally, the position is on the right edge and vertically, the height is randomly
            # sampled from the set of permissible values
            bird_x = self.x_max
            bird_y = random.randrange(self.y_min, self.y_max)
            spawned_bird.set_position(self.x_max, bird_y)

            # Append the spawned bird to the elements currently present in Env.
            self.elements.append(spawned_bird)

            # Spawn a fuel at the bottom edge with prob 0.01
        if random.random() < 0.01:
            # Spawn a fuel tank
            spawned_fuel = Fuel("fuel".format(self.bird_count), self.x_max, self.x_min, self.y_max, self.y_min)
            self.fuel_count += 1

            # Compute the x,y co-ordinates of the position from where the fuel tank has to be spawned
            # Horizontally, the position is randomly chosen from the list of permissible values and
            # vertically, the position is on the bottom edge
            fuel_x = random.randrange(self.x_min, self.x_max)
            fuel_y = self.y_max
            spawned_fuel.set_position(fuel_x, fuel_y)

            # Append the spawned fuel tank to the elemetns currently present in the Env.
            self.elements.append(spawned_fuel)

            # For elements in the Ev
        for elem in self.elements:
            if isinstance(elem, Bird):
                # If the bird has reached the left edge, remove it from the Env
                if elem.get_position()[0] <= self.x_min:
                    self.elements.remove(elem)
                else:
                    # Move the bird left by 5 pts.
                    elem.move(-5, 0)

                # If the bird has collided.
                if self.has_collided(self.chopper, elem):
                    # Conclude the episode and remove the chopper from the Env.
                    done = True
                    reward = -100

                    self.elements.remove(self.chopper)

            if isinstance(elem, Fuel):
                # If the fuel tank has reached the top, remove it from the Env
                if elem.get_position()[1] <= self.y_min:
                    self.elements.remove(elem)
                else:
                    # Move the Tank up by 5 pts.
                    elem.move(0, -5)

                # If the fuel tank has collided with the chopper.
                if self.has_collided(self.chopper, elem):
                    # Remove the fuel tank from the env.
                    if self.elements.__contains__(elem):
                        self.elements.remove(elem)

                    # Fill the fuel tank of the chopper to full.
                    self.fuel_left = self.max_fuel

        # Increment the episodic return
        self.ep_return += 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # If out of fuel, end the episode.
        if self.fuel_left == 0 and self.total_reward > 5000:
            done = True

        # actual_state = {}
        # for e in self.elements:
        #     actual_state[abs(e.x - self.chopper.x) + abs(e.y - self.chopper.y)] = (e.x, e.y)

        # print(np.argmin(actual_state))
        return self.canvas, reward, done, []

    def reset(self):
        # Reset the fuel consumed
        self.fuel_left = self.max_fuel

        # Reset the reward
        self.ep_return = 0

        # Reset the total_rewards
        self.total_reward = 0
        # Number of birds
        self.bird_count = 0
        self.fuel_count = 0

        # Determine a place to intialise the chopper in
        x = random.randrange(int(self.observation_shape[0] * 0.05), int(self.observation_shape[0] * 0.10))
        y = random.randrange(int(self.observation_shape[1] * 0.15), int(self.observation_shape[1] * 0.20))

        # Intialise the chopper
        self.chopper = Chopper("chopper", self.x_max, self.x_min, self.y_max, self.y_min)
        self.chopper.set_position(x, y)

        # Intialise the elements
        self.elements = [self.chopper]

        # Reset the Canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # return the observation
        # actual_state = []
        # for e in self.elements:
        #     actual_state.append((e.name, e.x, e.y))

        return self.canvas

    def render(self, mode="human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()

    def get_action_meanings(self):
        return {0: "Right", 1: "Left", 2: "Down", 3: "Up"}


def build_model(state_space_shape, num_actions, learning_rate):
    model = Sequential()

    model.add(Flatten(input_shape=state_space_shape))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))

    model.compile(adam_v2.Adam(learning_rate=1e-3), loss=MeanSquaredError())

    print(model.summary())
    return model


def build_conv2d_model(state_space_shape, num_actions, learning_rate):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(state_space_shape), activation='relu'))
    model.add(Flatten(state_space_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))

    model.compile(adam_v2.Adam(learning_rate=1e-3), loss=MeanSquaredError())
    print(model.summary())
    return model


def build_keras_model(state_size, num_actions):
    input = Input(shape=(1, state_size))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(num_actions, activation='linear')(x)
    model = Model(input, output)
    print(model.summary())
    model.compile(adam_v2.Adam(learning_rate=1e-3), metrics=['mae'])
    return model
