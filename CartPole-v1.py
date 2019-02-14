from collections import namedtuple
from time import sleep

import tensorflow as tf
import numpy as np
import gym
import time

from tensorflow.keras.models import Sequential # pylint: disable=import-error
from tensorflow.keras.layers import Dense # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam # pylint: disable=import-error

ENV_NAME = 'CartPole-v1'

EPOCHS = 100

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 32

EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = np.empty(self.capacity, dtype=Transition)
        self.current = 0
    
    def push(self, *args):
        self.memory[self.current] = Transition(*args)
        self.current = (self.current + 1) % self.capacity
        
    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)
    
    def __len__(self):
        return len([i for i, x in enumerate(self.memory) if x != None])


### import  the environnement ###
env = gym.make(ENV_NAME)
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]

replay_memory = ReplayMemory(MEMORY_SIZE)
epsilon = EPSILON_MAX

### Define the model ###
model = Sequential([
    Dense(32, input_shape=(4,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(action_space, activation='linear')
])

model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

start_time = time.time()


for _ in range(EPOCHS):
    observation = env.reset()
    state = np.reshape(observation, [1, observation_space])
    done = False
    step = 0

    while not done:
        if np.random.uniform(0.0, 1.0) < epsilon:
            action = np.random.randint(action_space)
        else:
            with tf.device('/gpu:0'):
                q_value = model.predict(state)
            action = np.argmax(q_value[0])

        observation, reward, done, info = env.step(action)
        reward = reward if not done else -reward

        next_state = np.reshape(observation, [1, observation_space])
        replay_memory.push(state, action, next_state, reward)
        
        if len(replay_memory) > BATCH_SIZE:
            samples = replay_memory.sample(BATCH_SIZE)

        
        env.render()

        epsilon = max(EPSILON_MIN, epsilon*EPSILON_DECAY)
        step += 1
        if done:
            print("step: ",step)
            break

print("--- %s seconds ---" % (time.time() - start_time))
