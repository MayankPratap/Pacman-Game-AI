# This version plays randomly.

import gym
import random
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.classic_control import rendering


def repeat_upsample(rgb_array, k=1, l=1, err=[]):
	# repeat kinda crashes if k/l are zero
	if k <= 0 or l <= 0: 
		if not err: 
			print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
			err.append('logged')
		return rgb_array

	# repeat the pixels k times along the y axis and l times along the x axis
	# if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

	return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


viewer = rendering.SimpleImageViewer()


env=gym.make('MsPacman-ram-v0')

episodes=1000

max_score=0
sum_scores=0  # Which will keep sum of scores achieved in episodes
rewards=[]
			#testing_scores=[]
mean_rewards=[]

for episode in range(episodes):

	done=False

	env.reset()

	episode_reward=0

	while not done:
		

		rgb = env.render('rgb_array')
		upscaled=repeat_upsample(rgb,4, 4)
		viewer.imshow(upscaled)


		action=env.action_space.sample()

		next_state,reward,done,info=env.step(action)

		episode_reward+=reward

		state=next_state


	rewards.append(episode_reward)

	sum_scores+=episode_reward

	mean_rewards.append(sum_scores*1.0/(episode+1))


	print("Episode {}# Score: {} Mean reward: {}".format(episode+1,episode_reward,mean_rewards[-1]))

	if episode_reward>=max_score:
		#print("Got a high score.")
		max_score=episode_reward

env.render(close=True)


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(episode+1),np.array(mean_rewards),label="Average scores over episodes")
plt.title("Episodes and average scores")
plt.xlabel("Episode #")
plt.ylabel("Average scores")
plt.savefig("./pacmanmean.png")

#env.render(close=True)
