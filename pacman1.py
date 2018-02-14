import gym
import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD,Adam 
import matplotlib.pyplot as plt 



class Player():

	def __init__(self,state_size,action_size):
		self.weights="./pacman.h5"
		self.state_size=state_size
		self.action_size=action_size
		self.memory=deque(maxlen=10000)
		self.learning_rate=0.0002
		self.gamma=0.95  
		self.exploration_rate=1.0
		self.exploration_min=0.1
		self.exploration_decay=0.0000009  # This will be decreased from epsilon at each time step
		self.replay_start=100
		#self.testing_states=deque(maxlen=10000)  # A deque containing some random states which we will play on to see our average scores over episodes
		self.model=self.build_model()

	def build_model(self):
		# Neural Network architecture
		model = Sequential()
		# I have to try different activation functions here.
		model.add(Dense(128, input_dim=self.state_size, activation='relu')) # First hidden layer
		#model.add(Dropout(0.2))
		model.add(Dense(128, activation='relu'))                            # 2nd hidden layer
		#model.add(Dropout(0.2))
		model.add(Dense(self.action_size, activation='linear'))				# Output layer
		model.summary()
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))    

		if os.path.isfile(self.weights):
			model.load_weights(self.weights)
			self.exploration_rate = self.exploration_min
		return model

	def save_model(self):
			self.model.save_weights(self.weights)

	def act(self, state):
		if np.random.rand()<=self.exploration_rate:
			return random.randrange(self.action_size)
		else:
			act_values=self.model.predict(state)
			return np.argmax(act_values[0])

	def remember(self, state, action, reward, next_state, done):
		
		self.memory.append((state, action, reward, next_state, done))
		if self.exploration_rate>self.exploration_min:
			self.exploration_rate-=self.exploration_decay

		#print("Modified exploration rate is : {}".format(self.exploration_rate))
		

	def replay(self, sample_batch_size):
		if len(self.memory)<self.replay_start: # Unless memory is filled with 100 entries dont start training
			return

		x_batch,y_batch=[],[]
		minibatch=random.sample(self.memory,min(len(self.memory),sample_batch_size))

		for state,action,reward,next_state,done in minibatch:
			y_target=self.model.predict(state)
			y_target[0][action]=reward if done else reward+self.gamma*np.max(self.model.predict(next_state)[0])
			x_batch.append(state[0])
			y_batch.append(y_target[0])
		self.model.fit(np.array(x_batch),np.array(y_batch),batch_size=len(x_batch),verbose=0)

class Pacman:
	def __init__(self):
		self.sample_batch_size=32
		self.episodes=10000
		self.env=gym.make('MsPacman-ram-v0')
		self.state_size=self.env.observation_space.shape[0]
		self.action_size=self.env.action_space.n
		self.player=Player(self.state_size, self.action_size)

	def prepare(self):

		for episode in range(50):

			state=self.env.reset()
			state=np.reshape(state,[1,self.state_size])

			self.player.testing_states.append(state)

			done=False
			step=0
			episode_reward=0

			while not done:
				action=self.player.act(state)
				next_state,reward,done,info=self.env.step(action)
				next_state=np.reshape(next_state, [1, self.state_size])
				state=next_state
				self.player.testing_states.append(state)

		print("Done preparing testing states. Hail Mayank, the Lord of Deep Learning. :P ")

	def learn(self):

		try:
			max_score=0

			rewards=[]
			testing_scores=[]

			mean_rewards=[]
			median_rewards=[]

			for episode in range(self.episodes):

				

				# I will use a frame skip of size 4
				frame_counter=0

				state=self.env.reset()
				state=np.reshape(state, [1, self.state_size])

				done=False
				step=0
				episode_reward=0   # Total reward in an episode
				#cur_lives=3 

				while not done:
					self.env.render()

					if frame_counter%4==0:  # for first frame in 4 frames we predict action
						action=self.player.act(state)


					next_state,reward,done,info=self.env.step(action)

					#if info['ale.lives']<cur_lives: # Someone made pacman dead
					#	reward=-100                 # Get immediate reward of -100 if pacman is dead
					#	cur_lives=info['ale.lives']  # Decrease cur_lives

	
					episode_reward+=reward

					next_state=np.reshape(next_state, [1, self.state_size])

					if frame_counter%4==3: # Only for last frame we remember the action
						self.player.remember(state, action, reward, next_state, done)
						
					if done:  # If pacman becomes dead it is necessary to remember this. 
						self.player.remember(state, action, reward, next_state, done)
						

					state=next_state
					frame_counter+=1
					step+=1

				rewards.append(episode_reward)

				mean_rewards.append(np.mean(np.array(rewards)))
				median_rewards.append(np.median(np.array(rewards)))

				print("Episode {}# Score: {}".format(episode+1,episode_reward))

				# After every step do train from memory
				print("Training after {} episode.".format(episode+1))
				self.player.replay(self.sample_batch_size)

				
				print("Exploration rate after episode {} is {}".format(episode+1,self.player.exploration_rate))

				if episode_reward>=max_score:
					#print("Got a high score.")
					max_score=episode_reward

				if episode%50==0:
					self.player.save_model()

				plt.style.use("ggplot")
				plt.figure()
				plt.plot(np.arange(episode+1),np.array(mean_rewards),label="Average score")
				plt.title("Episodes and average rewards")
				plt.xlabel("Episode #")
				plt.ylabel("Average rewards")
				plt.savefig("./pacmanmean.png")


				plt.figure()
				plt.plot(np.arange(episode+1),np.array(median_rewards),label="Median score")
				plt.title("Episodes and median rewards")
				plt.xlabel("Episode #")
				plt.ylabel("Median rewards")
				plt.savefig("./pacmanmedian.png")

				#print("Now I am going to test my model on testing state...")
			
				# After each episode I will test my model on testing states.
				avg_score=0   # Avg of maximum cumulative scores I can get on my testing states

				"""

				for test_state in self.player.testing_states:
					act_values=self.player.model.predict(test_state)
					avg_score+=np.amax(act_values[0])

				avg_score=avg_score/len(self.player.testing_states)

				print("Average score of all testing states is {}".format(avg_score))

				testing_scores.append(avg_score)

				"""



			print("Highest score obtained during training is: {}".format(max_score))		

			

			"""

			

			"""

	
			

		finally:
			self.player.save_model()
			self.env.render(close=True)
			
	def play(self):

		for episode in range(self.episodes):

			state=self.env.reset()
			state=np.reshape(state,[1,self.state_size])
			done=False
			step=0

			episode_reward=0   # Total reward in an episode

			while not done:
				self.env.render()
				action=self.player.act(state)
				next_state,reward,done,_=self.env.step(action)
				episode_reward+=reward
				next_state=np.reshape(next_state,[1,self.state_size])
				state=next_state
				step+=1

			print("Episode {}# Score: {}".format(episode,episode_reward))

		self.env.render(close=True)


if __name__ == "__main__":
	pacman = Pacman()
	#pacman.prepare()
	pacman.learn()
