from pong import Pong
import cv2 as cv
from agent import Agent
import config
import numpy as np
import tensorflow as tf
# 12k epochs
def pickle_vars(variables, file_name):
	import pickle
	for var in variables:
		with open(file_name, 'wb+') as file:
			pickle.dump(variables, file)

def load_vars(file_name):
	import pickle
	with open(file_name, 'rb') as file:
		return pickle.load(file)

def sample_action(alpha, q_values):	 # alpha is the probability with which the agent takes a random action
	if(np.random.uniform() < alpha):
		return np.random.randint(q_values.shape[-1])
	else:
		return tf.argmax(q_values, axis = 1).numpy().item()

def run(agents, pong):
	pong.reset()
	for agent in agents:
		agent.reset()

	rewards = 0.0
	done = False
	total_reward_l, total_reward_r = 0.0, 0.0
	while not done:
		state_a = [np.expand_dims(pong.get_screen(), axis = -1)]
		state_a = [np.expand_dims(pong.get_screen(), axis = -1)]
		q_values_a = agents[0].forward_prop(state_a)
		q_values_b = agents[1].forward_prop(state_b)

		# now get the next state, run this info through the agents and remove the preivous to previous screen from states vairable
		action_a = tf.argmax(q_values_a, axis = 1)[0]
		action_b = tf.argmax(q_values_b, axis = 1)[0]
		pong.render()
		reward_l, reward_r, done = pong.step(action_a.numpy(), action_b.numpy())
		total_reward_l += reward_l
		total_reward_r += reward_r
	return total_reward_l, total_reward_r, pong.hits

def train_episode(agents, target_agents, pong):
	pong.reset()
	for agent in agents + target_agents:
		agent.reset()
	state_a = [np.expand_dims(pong.get_screen(), axis = -1)]
	state_a = [np.expand_dims(pong.get_screen(), axis = -1)]

	_ = target_agents[0].forward_prop(state_a)
	_ = target_agents[1].forward_prop(state_b)

	rewards = 0.0
	total_reward_l, total_reward_r = 0.0, 0.0
	done = False
	time = 0
	loss = tf.constant(0.0)
	while not done:
		q_values_a = agents[0].forward_prop(state_a)
		q_values_b = agents[1].forward_prop(state_b)
		# now get the next state, run this info through the agents and remove the preivous to previous screen from states vairable
		action_a = sample_action(config.EXPLORE_PROB, q_values_a)
		action_b = sample_action(config.EXPLORE_PROB, q_values_b)
		pong.render()
		reward_l, reward_r, done = pong.step(action_a, action_b)
		total_reward_l += reward_l
		total_reward_r += reward_r
		rewards += reward_l + reward_r
		if done:
			loss += tf.square(reward_l - q_values_a[0, action_a]) + tf.square(reward_r - q_values_b[0, action_b])
		else:
			state_a = [np.expand_dims(pong.get_screen(), axis = -1)]
			state_a = [np.expand_dims(pong.get_screen(), axis = -1)]

			target_q_values_a = target_agents[0].forward_prop(state_a)
			target_q_values_b = target_agents[1].forward_prop(state_b)
			loss += tf.square(reward_l + config.DISCOUNT_FACTOR * tf.reduce_max(target_q_values_a) - q_values_a[0, action_a]) + tf.square(reward_r + config.DISCOUNT_FACTOR * tf.reduce_max(target_q_values_b) - q_values_b[0, action_b])

	return total_reward_l, total_reward_r, pong.hits, loss

def update_target_network(agents, target_agents):
	for target_agent, agent in zip(target_agents, agents):
		target_agent.set_weights(agent.get_weights())

def main():
	description = 'no_par_shar lr: {}, exp_prob: {}'.format(config.LEARNING_RATE, config.EXPLORE_PROB)
	pong = Pong(description)
	
	agents = [Agent(3), Agent(3)]
	target_agents = [Agent(3), Agent(3)]

	optimizer = tf.keras.optimizers.RMSprop(learning_rate = config.LEARNING_RATE)

	# Logger instances:
	import os
	os.system('rm -rf ./logs && mkdir ckpt')
	log_dir = './logs'
	summary_writer = tf.summary.create_file_writer(log_dir)
	os.system('tensorboard --logdir ./logs/ &')
	# to initialize, we run a random episode
	run(agents, pong)
	run(target_agents, pong)

	import sys
	if len(sys.argv) > 1:
		agents = load_vars(sys.argv[1])

	update_target_network(agents, target_agents)
	episodes = 0
	hit_log = []
	last_best = 0
	while True:
		with tf.GradientTape() as tape:
			reward_l, reward_r, time, loss = train_episode(agents, target_agents, pong)

		trainable_variables = agents[0].trainable_variables + agents[1].trainable_variables
		grads = tape.gradient(loss, trainable_variables)
		optimizer.apply_gradients(zip(grads, trainable_variables))
		episodes += 1
		hit_log.append(time)
		hit_log = hit_log[-30: ]
		if last_best < sum(hit_log) * 1.0 / len(hit_log):
			last_best = sum(hit_log) * 1.0 / len(hit_log)
			pickle_vars(agents, './ckpt/' + str(episodes) + '_' + str(last_best) + '.txt')

		with summary_writer.as_default():
			tf.summary.scalar('average-hits', sum(hit_log) * 1.0 / len(hit_log), episodes)
		print('Train Episode ({}) {:7d}:	Hits: {:4d}'.format(description, episodes, time))
		if episodes % config.EPISODES_TO_TRAIN == 0:
			reward_l, reward_r, time = run(agents, pong)
			print('Test Episode {:7d}:	Reward_l: {:7.4f}	Reward_r: {:7.4f}	Hits: {:4d}'.format(episodes, reward_l, reward_r, time))
			update_target_network(agents, target_agents)


if __name__ == '__main__':
	main()