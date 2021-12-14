import gym, matplotlib.pyplot as plt
from policy_network import *

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time
env = gym.make('LunarLander-v2')
running_reward = 0
action_number = 4

RL = PolicyGradient(action_number, env.observation_space.shape[0])

for episode in range(3000):
	observation = env.reset()
	while True:
		if RENDER: env.render()
		action = RL.choose_action(observation)
		observation_, reward, done, info = env.step(action)
		RL.store_transition(observation, action, reward)

		if done:
			ep_rs_sum = sum(RL.ep_rs)

			if 'running_reward' not in globals():
				running_reward = ep_rs_sum
			else:
				running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
			if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
			print('episode:', episode, '  reward:', int(running_reward))
			vt = RL.learn()
			if episode == 0:
				plt.plot(vt)  # plot the episode rewards
				plt.xlabel('episode steps')
				plt.ylabel('normalized state-action value')
				plt.show()
				plt.close()
			break
		observation = observation_