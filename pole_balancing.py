import os
from random import sample
from copy import deepcopy
from collections import deque
import tensorflow as tf
import gym
from time import sleep 
import numpy as np

def get_logfile():
    if not os.path.exists('logs/'):
        os.mkdir('logs/')
        dir = 'logs/1'
        os.mkdir(dir)
        return dir
    else:
        l = os.listdir('logs/');
        l=max([int(x) for x in l])
        n = l+1
        return 'logs/'+str(n)

def weight_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=name)

def bias_variable(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name)

# Initialize network
env = gym.make('CartPole-v1') 
env.reset()
x = tf.placeholder(tf.float32, [1,4],name="x")
y_ = tf.placeholder(tf.float32, [2],name="yhat")
score = tf.placeholder(tf.float32, name="score")
r_game = tf.placeholder(tf.float32, name="reward")
Q_game = tf.placeholder(tf.float32, name="Q")
tf.summary.scalar("score",score)
tf.summary.scalar("reward",r_game)
tf.summary.scalar("Q",Q_game)

with tf.name_scope("fully_connect_1"):
    W_fc1 = weight_variable([4, 64], "W_fc1")
    b_fc1 = bias_variable([64], "b_fc1")
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
    tf.summary.histogram("weights", W_fc1)
    tf.summary.histogram("biases", b_fc1)
    tf.summary.histogram("activations", h_fc1)
    tf.add_to_collection('vars', W_fc1)
    tf.add_to_collection('vars', b_fc1)

with tf.name_scope("fully_connect_2"):
    W_fc2 = weight_variable([64, 2], "W_fc2")
    b_fc2 = bias_variable([2], "b_fc2")
    y_net = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    tf.summary.histogram("weights", W_fc2)
    tf.summary.histogram("biases", b_fc2)
    tf.summary.histogram("activations", y_net)
    tf.add_to_collection('vars', W_fc2)
    tf.add_to_collection('vars', b_fc2)


# Setup loss function
nextQ = tf.placeholder(shape=[2],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - y_net))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

# Set learning parameters
gamma = .8
e = 1
num_episodes = 2000
logfile = get_logfile()
ms = tf.summary.merge_all()
writer = tf.summary.FileWriter(logfile)

def replay(memory,sess,y):
    print "memorizing..."
    mem_sample = sample(memory,min(200,len(memory)))
    for s,a,r,ns,d in mem_sample:
        targetQ = sess.run(y_net,feed_dict={x:s})
        Q1 = sess.run(y_net,feed_dict={x:ns})
        targetQ[0][a] = r + y*max(Q1[0])
        targetQ=targetQ[0]
        sess.run(updateModel,feed_dict={x:s,nextQ:targetQ})


with tf.Session() as sess:
    writer.add_graph(sess.graph)
    sess.run(init) 
    itrs = []
    scores = []
    memory = deque(maxlen=5000)
    for i_episode in range(num_episodes):
        observation = env.reset()
        action = 0
        rewards = []
        for t in range(300):
            env.render()
            obs = observation.reshape((1,4))
            Q=sess.run(y_net,feed_dict={x:obs})
            action = np.argmax(Q[0])
            observation, reward, done, info = env.step(action)
            Qr=deepcopy(Q)
            reward = abs(obs[0][-1])
            rewards.append(reward)
            Qr[0][action] = reward+gamma*Q[0][action]
            Qr=Qr[0]
            # print action
            # print Q[0], "->", Qr, reward
            # print obs
            sess.run(updateModel,feed_dict={x:obs,nextQ:Qr})
            s = sess.run(ms,feed_dict={x:obs, score:t, r_game:reward, Q_game:1})
            ns = observation.reshape((1,4))
            if t>50:
                memory.append((obs,action,reward,ns,done))
            writer.add_summary(s,t)
            if done:
                replay(memory,sess,gamma)
                if t<10:
                    sess.run(init)
                print("Episode {} finished after {} timesteps".format(i_episode,t+1))
                print sum(rewards)
                break
