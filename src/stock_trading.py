"""
Use DDPG to train a stock trader based on a window of history price
"""

from __future__ import print_function, division

from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

from environment.portfolio import PortfolioEnv
from utils.data import read_stock_history, normalize

import numpy as np
import tflearn
import tensorflow as tf
import argparse
import pprint

DEBUG = True

def get_batch_norm_str(use_batch_norm):
    return 'batch_norm' if use_batch_norm else 'no_batch_norm'

def get_model_path(window_length, predictor_type, use_batch_norm):
    return 'weights/stock/{}/window_{}/{}/checkpoint.ckpt'.format(predictor_type, window_length,
                                                                  get_batch_norm_str(use_batch_norm))


def get_result_path(window_length, predictor_type, use_batch_norm):
    return 'results/stock/{}/window_{}/{}/'.format(predictor_type, window_length,
                                                   get_batch_norm_str(use_batch_norm))


def get_variable_scope(window_length, predictor_type, use_batch_norm):
    return '{}_window_{}_{}'.format(predictor_type, window_length,
                                    get_batch_norm_str(use_batch_norm))


def stock_predictor(inputs, predictor_type, use_batch_norm):
    """This the deep neuro network for policy gradient
    TODO: change this to use keras in TF
    """
    window_length = inputs.get_shape()[2]
    assert predictor_type in ['cnn', 'lstm'], 'type must be either cnn or lstm'
    if predictor_type == 'cnn':
        net = tflearn.conv_2d(inputs, 32, (1, 3), padding='valid')
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.conv_2d(net, 32, (1, window_length - 2), padding='valid')
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        if DEBUG:
            print('After conv2d:', net.shape)
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)
    elif predictor_type == 'lstm':
        num_stocks = inputs.get_shape()[1]
        hidden_dim = 32
        net = tflearn.reshape(inputs, new_shape=[-1, window_length, 4]) #changed 1 to 4 for default normalizer
        if DEBUG:
            print('Reshaped input:', net.shape)
        net = tflearn.lstm(net, hidden_dim)
        if DEBUG:
            print('After LSTM:', net.shape)
        net = tflearn.reshape(net, new_shape=[-1, num_stocks, hidden_dim])
        if DEBUG:
            print('After reshape:', net.shape)
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)
    else:
        raise NotImplementedError

    return net


class StockActor(ActorNetwork):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        ActorNetwork.__init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size)

    def create_actor_network(self):
        """
        self.s_dim: a list specifies shape
        """
        print("####", self.s_dim)
        nb_classes, window_length, features = self.s_dim
        assert nb_classes == self.a_dim[0]
        assert window_length > 2, 'This architecture only support window length larger than 2.'
        inputs = tflearn.input_data(shape=[None] + self.s_dim, name='input')

        net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm)

        net = tflearn.fully_connected(net, 64)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 64)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim[0], activation='softmax', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        window_length = self.s_dim[1]
        # print("inputs' shape={}".format(inputs.shape))
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })


class StockCritic(CriticNetwork):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars,
                 predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        CriticNetwork.__init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None] + self.s_dim)
        action = tflearn.input_data(shape=[None] + self.a_dim)

        net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 64)
        t2 = tflearn.fully_connected(action, 64)

        net = tf.add(t1, t2)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })


def obs_normalizer(observation):
    """ Preprocess observation obtained by environment

    Args:
        observation: (nb_classes, window_length, num_features) or with info

    Returns: normalized

    """
    # print("data before normalization={}".format(observation))
    if isinstance(observation, tuple):
        observation = observation[0]
    # print("data in first element used={}".format(observation))
    # directly use close/open ratio as feature
    # print("shape before normalization={}".format(observation.shape))
    observation = observation[:, :, 3:4] / observation[:, :, 0:1]
    observation = normalize(observation)
    # print("shape after normalization={}".format(observation.shape))
    return observation

def default_normalizer(observation):
    """ Preprocess observation obtained by environment

    Args:
        observation: (nb_classes, window_length, num_features) or with info

    Returns: make sure the data is numpy array, nothign else

    """
    # print("data before normalization={}".format(observation))
    if isinstance(observation, tuple):
        observation = observation[0]
    # print("data in first element used={}".format(observation))
    # directly use close/open ratio as feature
    # print("shape before normalization={}".format(observation.shape))
    observation = observation / observation[:, 0:1, :]
    # observation = normalize(observation)
    # print("data in first element used={}".format(observation))
    # print("shape after normalization={}".format(observation.shape))
    return observation

def test_model(env, model):
    observation, info = env.reset()
    print("first observation={}".format(observation))
    #TODO dimensions not matched, reson: model is using the normalization function which only take the close/open
    # as the feature and ignore others so the dimension is 1 feature instaed of 4
    # observation = observation[:, :, 3] / observation[:, :, 0]
    # print("observation after normalization=", observation)
    # observation = np.expand_dims(observation, axis=-1)
    # print("observation after dims expand=", observation)

    done = False
    step = 1
    while not done:
        action = model.predict_single(observation)
        print("action at step {}={}".format(step, action))
        step += 1
        observation, _, done, _ = env.step(action)
    env.render()


def test_model_multiple(env, models):
    observation, info = env.reset()
    done = False
    while not done:
        actions = []
        for model in models:
            actions.append(model.predict_single(observation))
        actions = np.array(actions)
        observation, _, done, info = env.step(actions)
    env.render()
    env.plot()

def _load_model(norm_func=None):
    ddpg_model = DDPG(env, sess, actor, critic, actor_noise, obs_normalizer=norm_func,
                      config_file='config/stock.json', model_save_path=model_save_path,
                      summary_path=summary_path)
    ddpg_model.initialize(load_weights=True)
    return ddpg_model

def predict_next_day(env, sess, actor, critic, actor_noise, norm_func=None):
    ddpg_model = DDPG(env, sess, actor, critic, actor_noise, obs_normalizer=norm_func,
                      config_file='config/stock.json', model_save_path=model_save_path,
                      summary_path=summary_path)
    ddpg_model.initialize(load_weights=True)
    env = PortfolioEnv(last_history, target_stocks, steps=0, window_length=window_length, start_idx=0, trading_cost=0.0, sample_start_date='2019-12-26')
    print("data=", last_history)
    observation = env.get_last_observation()
    # print("observation before normalization={}, shape={}".format(observation, observation.shape))
    # observation = observation[:, :, 3] / observation[:, :, 0]
    # print("observation after normalization={}, shape={}".format(observation, observation.shape))
    # observation = np.expand_dims(observation, axis=-1)
    # print("observation after dims expand={}, shape={}".format(observation, observation.shape))
    action = ddpg_model.predict_single(observation)
    # action = np.squeeze(action, axis=0)
    # observation, _, done, _ = env.step(action)
    print("action=", action)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Provide arguments for training different DDPG models')

    parser.add_argument('--debug', '-d', help='print debug statement', default=False)
    parser.add_argument('--predictor_type', '-p', help='cnn or lstm predictor', required=True)
    parser.add_argument('--window_length', '-w', help='observation window length', required=True)
    parser.add_argument('--batch_norm', '-b', help='whether to use batch normalization', required=True)
    parser.add_argument('--train', '-t', help='whether to train or to predict', required=True)
    parser.add_argument('--test', '-T', help='whether to test the saved model', required=False)
    parser.add_argument('--obs', '-o', help='whether to use close open ratio normalizer', required=False)

    args = vars(parser.parse_args())

    pprint.pprint(args)

    if args['debug'] == 'True':
        DEBUG = True
    else:
        DEBUG = False

    if args['train'] == 'True':
        TRAIN = True
    else:
        TRAIN = False

    if args['test'] == 'True':
        TEST = True
    else:
        TEST = False

    #data dimensions: ticker:price date:open high low close volume
    # history, tickers = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
    history, tickers = read_stock_history(filepath='c:/data/equity/price/target_prices.h5')
    if DEBUG:
        print("all data before slicing=", history)
    print("all data's shape={}".format(history.shape))
    history = history[:, :, :4]
    if DEBUG:
        print("data=", history[0], "\nshape=", history.shape, "\n stocks=", tickers)
        print("all data=", history)
    target_stocks = tickers
    print("stocks in scope={}".format(tickers))
    #the set of prices at the begining for training, the rest for validation, 1825 in total for this data set
    # num_training_time = history.shape[1] #2095
    num_training_time = 3000
    if TEST:
        num_training_time = history.shape[1]

    window_length = int(args['window_length'])
    nb_classes = len(target_stocks) + 1

    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[tickers.index(stock), :num_training_time, :]

    #last data point - for prediction
    last_history = np.empty(shape=(len(target_stocks), window_length+1, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        last_history[i] = history[tickers.index(stock), -window_length-1:, :]

    # setup environment
    env = PortfolioEnv(target_history, target_stocks, steps=1000, window_length=window_length)

    if DEBUG:
        print('target_history shape=', target_history.shape)

    action_dim = [nb_classes]
    state_dim = [nb_classes, window_length]
    batch_size = 64 #TODO this should be from the config
    if args['obs'] == 'True':
        NORM_FUNC = obs_normalizer
        state_dim += [1]
    else:
        NORM_FUNC = default_normalizer
        state_dim += [4]

    action_bound = 1.
    tau = 1e-3
    assert args['predictor_type'] in ['cnn', 'lstm'], 'Predictor must be either cnn or lstm'
    predictor_type = args['predictor_type']
    if args['batch_norm'] == 'True':
        use_batch_norm = True
    elif args['batch_norm'] == 'False':
        use_batch_norm = False
    else:
        raise ValueError('Unknown batch norm argument')
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    model_save_path = get_model_path(window_length, predictor_type, use_batch_norm)
    summary_path = get_result_path(window_length, predictor_type, use_batch_norm)

    variable_scope = get_variable_scope(window_length, predictor_type, use_batch_norm)

    with tf.variable_scope(variable_scope):
        sess = tf.Session()
        actor = StockActor(sess, state_dim, action_dim, action_bound, 1e-4, tau, batch_size,
                           predictor_type, use_batch_norm)
        critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=1e-3,
                             learning_rate=1e-3, num_actor_vars=actor.get_num_trainable_vars(),
                             predictor_type=predictor_type, use_batch_norm=use_batch_norm)

        if TRAIN:
            ddpg_model = DDPG(env, sess, actor, critic, actor_noise, obs_normalizer=NORM_FUNC,
                              config_file='config/stock.json', model_save_path=model_save_path,
                              summary_path=summary_path)
            ddpg_model.initialize(load_weights=False)
            ddpg_model.train(debug=DEBUG)
        elif TEST:
            env = PortfolioEnv(target_history, target_stocks, steps=(num_training_time - window_length - 2),
                               window_length=window_length)
            test_model(env, _load_model(norm_func=NORM_FUNC))
        else:
            # for prediction
            predict_next_day(env, sess, actor, critic, actor_noise, norm_func=NORM_FUNC)
