from keras.models import Sequential
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.layers import Dense, Flatten, Embedding, Reshape
from rl.agents import DQNAgent
from keras.optimizers import SGD, Adam, Adamax
from keras import backend as K

import gym

env = gym.make('FrozenLake-v0')
env.seed(123)

model = Sequential()
# model.add(Flatten(input_shape =((1,) + env.observation_space.shape)))
model.add(Embedding(16, 16, input_length=1, embeddings_initializer='identity'))
model.add(Reshape((16,)))
model.add(Dense(16, activation='linear'))
model.add(Dense(16, activation='linear'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(env.action_space.n))

memory = SequentialMemory(50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model = model, policy = policy, memory = memory, nb_actions=env.action_space.n, target_model_update= 1e-2)
dqn.compile(optimizer=SGD(), metrics=['mae','accuracy'])
dqn.fit(env, nb_steps=5000, visualize=False, verbose=True)
dqn.test(env, nb_episodes=10, visualize=False, verbose=True)
