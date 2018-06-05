'''
Created on 2016/02/19

@author: takuya-hv2
'''
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
import tf_util as U
import argparse
import copy
from replay_buffer import ReplayBuffer
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for stochastic games")
    parser.add_argument("--agent", type =str, default = 'I3', help = "agent type")
    parser.add_argument("--game", type=str, default="prison", help="name of the scenario script")
    parser.add_argument("--timestep", type= int, default= 2, help ="the time step of act_trajectory")
    parser.add_argument("--iteration", type=int, default=60000, help="number of episodes")
    parser.add_argument("--seed", type = int, default = 13, help = "random seed")
    parser.add_argument("--episodes", type = int, default = 100, help = "episodes")
    parser.add_argument("--steps", type = int, default = 1000, help ="steps in one episode" )
    parser.add_argument("--batch_size", type = int, default = 200, help = "set the batch_size")
    parser.add_argument("--warm_up_steps", type = int, default = 1000, help = "set the warm up steps")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=8, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):

        out1 = layers.fully_connected(input, num_outputs=num_units, activation_fn=tf.nn.relu)
        # out2 = layers.fully_connected(out1, num_outputs=num_units, activation_fn=tf.nn.relu)
        out3 = layers.fully_connected(out1, num_outputs=num_outputs)

        return out3


def lstm_model(input, num_outputs, scope, reuse = False, num_units = 4):

    with tf.variable_scope(scope, reuse = reuse):
        weight = tf.get_variable("W", shape=[num_units, num_outputs],
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("B", shape=[num_outputs],
                               initializer=tf.contrib.layers.xavier_initializer())
        # x = tf.reshape(input, (-1, TIMESTEPS, input.get_shape().as_list()[1] *input.get_shape().as_list()[-1]))
        x = tf.unstack(input, axis=1)
        rnn_cell = rnn.BasicLSTMCell(int(num_units))
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32, scope="trainging_agent")
        out = layers.fully_connected(outputs[-1], num_outputs=num_outputs, activation_fn=tf.nn.relu)
        # out =layers.fully_connected(outputs[-1], num_outputs=num_outputs)
        return out

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

class SimpleMatrixGame():
    def __init__(self, game):
        self.action_space=[0,1]
        if game == 'cooperative':
            self.payoff_1=[[10,-10],[-10,-10]]
            self.payoff_2=[[10,-10],[-10,-10]]
#     '''payoff matrix of zero-sumgame scenario. nash equilibrium: (Agenat1's action=0,Agent2's action=1)'''
        elif game == 'nash':
            self.payoff_1=[[5,2],[-1,6]]
            self.payoff_2=[[-5,-2],[1,-6]]
#     '''payoff matrix of zero-sumgame scenario. matching pennies'''
        elif game == 'zero_sum':
            self.payoff_1=[[1,-1],
                           [-1,1]]
            self.payoff_2=[[-1,1],
                           [1,-1]]
        elif game =='prison':
            self.payoff_1=[[-1,-3],
                           [0,-2]]
            self.payoff_2=[[-1,0],
                           [-3,-2]]

        self.reward_1=None
        self.reward_2 =None
    def step(self, actions):
        self.reward_1 = self.payoff_1[int(actions[0])][int(actions[1])]
        self.reward_2 =  self.payoff_2[int(actions[0])][int(actions[1])]
        return  [self.reward_1 , self.reward_2]
    def reset (self):
        self.reward_1=None
        self.reward_2 =None
    def get_joint_reward(self):
        return np.array(self.reward_1, self.reward_2)

class I3QLearner():
    def __init__(self, num_features, num_actions, timestep, action_space, scope):
        self.scope = scope
        self._lr = 0.5
        self.discount = 1.
        self.replay_buffer = ReplayBuffer(1e4)

        with tf.variable_scope(self.scope):
            self.act_trajectory = tf.placeholder(tf.float32, shape = ((None, timestep, action_space)))
            self.target = tf.placeholder(tf.float32, shape = ((None, )))
            self.act = tf.placeholder(tf.int32, shape = ((None,)))

            self.tau = lstm_model(self.act_trajectory, num_actions, scope = "tau_model_{}".format(scope))
            self.q_input = self.tau
            #train network
            self.q = mlp_model(self.q_input, 2, scope = "q_model_{}".format(scope))
            q_func_vars = U.scope_vars(U.absolute_scope_name( "q_model_{}".format(scope)))
            #target network
            self.target_q = mlp_model(self.q_input, 2, scope = "target_q_model_{}".format(scope))
            target_q_func_vars = U.scope_vars(U.absolute_scope_name( "target_q_model_{}".format(scope)))

            # take action
            self.softmax = tf.nn.softmax(self.target_q)
            self.pred = tf.argmax(self.softmax, axis = 1)

            #calculate the loss
            self.q_t_selected = tf.reduce_mean(self.q * tf.one_hot(self.act, num_actions), 1)
            q_tp1_best = tf.reduce_max(self.q, 1)
            q_tp1_best_masked =  q_tp1_best
            td_error = self.q_t_selected - tf.stop_gradient(self.target)
            self.errors = U.huber_loss(td_error)
            self.q_opt_op = tf.train.AdamOptimizer(self._lr).minimize(self.errors, var_list = q_func_vars)

            self.tau_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.tau, labels=self.act))
            self.tau_opt_op = tf.train.AdamOptimizer(self._lr).minimize(self.tau_loss)

            self.get_pred = U.function(inputs = [self.act_trajectory] , outputs = [self.softmax])
            self.train_q = U.function(inputs = [self.act_trajectory] + [self.target] +[self.act] , outputs = [self.errors, self.q], updates = [self.q_opt_op])
            self.train_tau = U.function(inputs =[ self.act] + [self.act_trajectory], outputs = [self.tau_loss], updates =[ self.tau_opt_op ])
            self.update_model = make_update_exp(q_func_vars, target_q_func_vars)

    def experience(self, action1, act_tra1 , reward1):
        self.replay_buffer.add(action1, act_tra1 , reward1)

    # bolzman exploration policy, set the temperature parmeter = 1 for default
    def get_act(self, act_trajectory):

        acpd = self.get_pred(act_trajectory)[0][0]
        # action = np.random.choice([0,1], p = acpd)
        action = epsilon_greedy(acpd, 0.1)
        return action

    def supervise_tau(self, a_next, action_trajectory):

        loss =  self.train_tau(*([a_next] + [action_trajectory]))[0]
        return loss
    def update_target(self):
        self.update_model()

    def learn(self, batch_size):

        replay_sample_index = self.replay_buffer.make_index(batch_size)
        act, act_tra, reward = self.replay_buffer.sample_index(replay_sample_index)
        loss , q= self.train_q(*([act_tra] + [reward] + [act]))
        return loss, q

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def epsilon_greedy(q_values, epsilon):
  if epsilon < np.random.random():
    return np.argmax(q_values)
  else:
    return np.random.randint(np.array(q_values).shape[-1])

class QLearner():
    def __init__(self, num_action):
        self._q = np.random.rand(num_action)
        self._lr = 0.001
        self.policy = epsilon_greedy(self._q, 0.1)
    def get_act(self):

        action =  np.argmax(self._q)
        return action

    def learn(self, a, r):
        self._q[a] += self._lr *r
        return self._q

class Frequency():
    def __init__(self, num_action):
        self.act0 = 0
        self.act1 = 0

    def get_act(self):
        return max(self.act0, self.act1)

    def learn(self, act):
        if act == 0:
            self.act0 += 1
        else:
            self.act1 += 1



import collections
import time
import random
import os

if __name__ == '__main__':
    arglist = parse_args()
    log_dir = './logs/{}'.format(arglist.game)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    action_space = 1
    if arglist.agent == "Freq":
        log = open('./logs/{}/{}_{}_step{}_seed_{}_{}.csv'.format(arglist.game, arglist.agent, time.time(), arglist.timestep, arglist.seed, n), 'w+')
        learner1 = QLearner(2)
        learner2 = QLearner(2)
        env =  SimpleMatrixGame(arglist.game)

        for i in range(arglist.iteration):
            action1 = learner1.get_act()
            action2 = learner2.get_act()
            reward_1,reward_2 = env.step([action1, action2])

            learner1.learn(action2)

    if arglist.agent == "Q":
        for n in range(10):
            random.seed(arglist.seed + n)
            log = open('./logs/{}/{}_{}_step{}_seed_{}_{}.csv'.format(arglist.game, arglist.agent, time.time(), arglist.timestep, arglist.seed, n), 'w+')
            step_log = open('./logs/{}/step_logs_{}_{}step{}_seed_{}_{}.csv'.format(arglist.game, arglist.agent, time.time(), arglist.timestep, arglist.seed, n), 'w+')
            env =  SimpleMatrixGame(arglist.game)
            learner1 = QLearner(2)
            learner2 = QLearner(2)
            for ep in range(arglist.episodes):
                reward1_list = np.array([])
                reward2_list = np.array([])
                for i in range(arglist.steps):
                    action1 = learner1.get_act()
                    action2 = learner2.get_act()

                    reward1, reward2 = env.step([action1, action2])
                    reward1_list = np.append(reward1_list, reward1 )
                    reward2_list = np.append(reward2_list, reward2 )
                    q1 = learner1.learn(action1, reward1)
                    q2 = learner2.learn(action2, reward2)
                    step_log.write('{}\n'.format(','.join(map(str, [action1,action2, reward1,reward2, q1[0], q1[1], q2[0], q2[1]]))))
                rew1_mean = np.mean(reward1_list)
                rew2_mean = np.mean(reward2_list)
                log.write('{}\n'.format(','.join(map(str, [rew1_mean, rew2_mean]))))

    elif arglist.agent == "I3":

        for n in range (10):
            tf.reset_default_graph()
            random.seed(int(arglist.seed + n))
            log = open('./logs/{}/batch_{}_{}_step{}_seed_{}_{}.csv'.format(arglist.game, arglist.agent, time.time(), arglist.timestep, arglist.seed, n), 'w+')
            step_log = open('./logs/{}/step_logs_{}_{}step{}_seed_{}_{}.csv'.format(arglist.game, arglist.agent, time.time(), arglist.timestep, arglist.seed, n), 'w+')
            with U.single_threaded_session():
                learner1 = I3QLearner(0, 2, arglist.timestep,action_space, "learner1")
                learner2 = I3QLearner(0, 2, arglist.timestep,action_space, "learner2")
                U.initialize()
                env = SimpleMatrixGame(arglist.game)
                global_step = 0
                turn = 0
                for ep in range(arglist.episodes):
                        #steping
                        episode_reward1 = np.array([])
                        episode_reward2 = np.array([])

                        action_trajectory_1 = collections.deque(np.zeros((arglist.timestep, action_space)), maxlen = arglist.timestep)
                        action_trajectory_2 = collections.deque(np.zeros((arglist.timestep, action_space)), maxlen = arglist.timestep)
                        nash_count = 0
                        turn = 1-turn
                        for step in range(arglist.steps):
                            global_step += 1
                            action1 = learner1.get_act([action_trajectory_2])
                            action2 = learner2.get_act([action_trajectory_1])


                            reward1, reward2 = env.step([action1, action2])

                            episode_reward1 = np.append(episode_reward1,reward1)
                            episode_reward2 = np.append(episode_reward2,reward2)

                            learner1.experience(action1, action_trajectory_2, reward1)
                            learner2.experience(action2, action_trajectory_1, reward2)
                            if global_step > arglist.warm_up_steps:
                                t1_loss = learner1.supervise_tau([action2], [action_trajectory_2] )
                                t2_loss = learner2.supervise_tau([action1], [action_trajectory_1])

                            #train
                            # if turn ==0:
                                p1_loss, p1_q = learner1.learn(arglist.batch_size)
                            # else:
                                p2_loss, p2_q = learner2.learn(arglist.batch_size)

                            # print(p1_loss, p2_loss, t1_loss, t2_loss)
                                if p1_q[0][0] == p1_q[0][1] == p2_q[0][0] == p2_q[0][1]:
                                    nash_count +=1

                                # learner1.update_target()
                                # learner2.update_target()
                                step_log.write('{}\n'.format(','.join(map(str, [action1,action2, reward1,reward2, p1_q[0][0], p1_q[0][1], p2_q[0][0] ,p2_q[0][1]]))))
                            else:
                                step_log.write('{}\n'.format(','.join(map(str, [action1, action2, reward1, reward2]))))

                            # print('{}\n'.format(','.join(map(str, [action1, action2, reward1, reward2, p1_q[0][0],
                            #                                            p1_q[0][1], p2_q[0][0], p2_q[0][1]]))))
                            action_trajectory_2.append(np.array([action1]))
                            action_trajectory_1.append(np.array([action2]))
                        # to set it update every step
                            # learner1.update_target()
                            # learner2.update_target()
                        # to set it update 500 steps

                        ep_mean_rew1 = np.mean(episode_reward1)
                        ep_mean_rew2 = np.mean(episode_reward2)


                        print( [ep_mean_rew1, ep_mean_rew2])
                        log.write('{}\n'.format(','.join(map(str, [ep_mean_rew1, ep_mean_rew2, nash_count]))))


