from gameenv import GameEnv
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
import gym

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

env = GameEnv(4)

def build_model(feature_dim, action_size, window_length=1):
    inputs = layers.Input(shape=(window_length, feature_dim))
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation=tf.nn.leaky_relu)(x)
    x = layers.Dense(128, activation=tf.nn.leaky_relu)(x)
    x = layers.Dense(128, activation=tf.nn.leaky_relu)(x)
    x = layers.Dense(64, activation=tf.nn.leaky_relu)(x)
    x = layers.Dense(64, activation=tf.nn.leaky_relu)(x)
    x = layers.Dense(64, activation=tf.nn.leaky_relu)(x)
    x = layers.Dense(64, activation=tf.nn.leaky_relu)(x)
    x = layers.Dense(64, activation=tf.nn.leaky_relu)(x)
    outputs = layers.Dense(action_size, activation='linear')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

states = env.observation_space.shape[0]
actions = env.action_space.n

model = build_model(states, actions, window_length=1)

model.summary()

def build_Agent(model, actions):
    # policy = BoltzmannQPolicy()
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10000, target_model_update=1e-2)
    return dqn

dqn = build_Agent(model, actions)
dqn.compile(optimizer=Adam(learning_rate=1e-5, clipnorm=1.0), metrics=['mae'])

dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)
