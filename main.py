import gym
import tensorflow as tf
import urllib3
import pygame
env = gym.make("CartPole-v1")
obs = env.reset()
import numpy as np
from tensorflow import keras

# Observation Space = (cart_pos, cart_velocity, angle, angular_velocity)
# Action Space = (0->Left, 1->Right)

input_size = tf.shape(tf.float32(obs[0]))
hidden_size = 4
output_size = 1

initi = tf.keras.initializers.VarianceScaling(
    scale=1.0,
    mode='fan_in',
    distribution='truncated_normal',
    seed=None
)

# Model
model = keras.models.Sequential()
model.add(keras.Input(name="observations", shape = [None, input_size], dtype=tf.float32))
model.add(keras.layers.Dense(hidden_size, dtype=tf.float32, initializer=initi, activation=tf.nn.relu))
logit = model.add(keras.layers.Dense(output_size, dtype=tf.float32, initialize=initi, acitvation=tf.nn.relu))
output = model.add(tf.nn.softmax) # Will have the probs


# Select action
action_space = keras.layers.concatenate(inputs = [output, 1-output], axis = 1)
action_space = tf.random.categorical(action_space, num_samples=1)


# Target Probability to calculate cross_entropy
y = 1. - tf.float32(action_space)
lr = 0.01
ce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logit)
optimizer = keras.optimizers.Adam(lr)
grads_vars = optimizer.compute_gradients(ce)

gradient_operation = optimizer.apply_gradients(grads_vars)


def cal_discount_rewards(rewards, discount: float):
    discounted = []
    discounted.insert(0, rewards[len(rewards) - 1])
    rolling_sum = discounted[0]
    for index_from_back in reversed(range(len(rewards) - 1)):
        rolling_sum = rewards[index_from_back] + rolling_sum * discount
        discounted.insert(0, rolling_sum)

    return np.asarray(discounted)

def normalized_rewards(all_rewards, discount:float):
    batch_rewards = [cal_discount_rewards(rewards, discount) for rewards in all_rewards]
    flat_rewards = np.concatenate(batch_rewards)
    mean_rewards = np.mean(flat_rewards)
    std_rewards = np.std(flat_rewards)
    normalized = [((rewards - mean_rewards)/ std_rewards) for rewards in batch_rewards]
    return normalized










