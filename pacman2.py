import gym
import random
import os
import pickle
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD,Adam 
import matplotlib.pyplot as plt 
from gym.envs.classic_control import rendering

# repeat_unsample function is used to resize the image screen.

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



# This code uses Target Network and Double Deep Q Learning
class Player():

	def __init__(self,state_size,action_size):
		self.weights="pacmangpu2c.h5" # This file saves the weights of trained model
		self.state_size=state_size	  # Memory of RAM state: 128 bytes
		self.action_size=action_size  # Total 9 actions possible
		self.memory=deque(maxlen=100000) # Replay memory
		self.learning_rate=0.0002        # Learning rate
		self.gamma=0.95  				 # Discounted future reward	
		self.exploration_rate=1.0       
		self.exploration_min=0.1
		self.exploration_decay=0.0000009  # This will be decreased from epsilon at each time step
		self.replay_start=100            
		#self.testing_states=deque(maxlen=10000)  # A deque containing some random states which we will play on to see our average scores over episodes
		self.model=self.build_model()  # Learning Model
		self.target_model=self.build_model() # Target Model

	def build_model(self):
		# Neural Network architecture
		model = Sequential()
		model.add(Dense(128, input_dim=self.state_size, activation='relu')) # 1st hidden layer
		#model.add(Dropout(0.1))															
		model.add(Dense(128, activation='relu'))							# 2nd hidden layer
		#model.add(Dropout(0.1))							
		model.add(Dense(128, activation='relu'))							# 3rd hidden layer
		#model.add(Dropout(0.1))
		model.add(Dense(128, activation='relu'))							# 4th hidden layer
		#model.add(Dropout(0.1))                                            					
		model.add(Dense(self.action_size, activation='linear'))				# Output layer
		model.summary()													
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

		if os.path.isfile(self.weights):   # This checks whether there is a trained model already present in current dir
			model.load_weights(self.weights)
			self.exploration_rate = 0.1
		return model


	def save(self):   # This function saves weights in a file.
			self.model.save_weights(self.weights)

	def act(self, state):  # This function chooses an action for a given state
		if np.random.rand()<=self.exploration_rate:
			return random.randrange(self.action_size)
		else:
			act_values=self.model.predict(state)
			return np.argmax(act_values[0]) # returns action

	def remember(self, state, action, reward, next_state, done):  # Save information in replay memory
		
		self.memory.append((state, action, reward, next_state, done))
		if self.exploration_rate>self.exploration_min:
			self.exploration_rate-=self.exploration_decay

		#print("Modified exploration rate is : {}".format(self.exploration_rate))
		

	def replay(self, sample_batch_size, special=None):  # Train the model using a random batch of replay memory.
		if len(self.memory)<self.replay_start: # Unless memory is filled with 100 entries dont start training
			return

		minibatch=random.sample(self.memory,min(len(self.memory),sample_batch_size))

		if special is not None:
			minibatch.append(special)


		for state,action,reward,next_state,done in minibatch:
			#print(self.model.predict(state)[0].shape)
			target=reward 
			if not done:
				n_action=np.argmax(self.model.predict(next_state)[0]) # Get action using learning model
				future_qvalue=self.target_model.predict(next_state)[0][n_action]
				target=reward+self.gamma*future_qvalue  # This is our target which we want my learning model to converge

			target_f=self.model.predict(state)   # Predict Q values for the given state 
			target_f[0][action]=target 	         # Change the target Q value we can get by this action
			self.model.fit(state,target_f,epochs=1,verbose=0)  # Set Q value 

	def target_train(self):   # Make target model same as learning model.
		model_weights=self.model.get_weights()
		self.target_model.set_weights(model_weights)

	def load_memory(self):    # Load replay memory from previous training.
		#memory=deque(maxlen=100000)
		with open('memorysaved.pkl','rb') as f:
			memory=pickle.load(f)
		return memory 

	def save_memory(self):    # Save replay memory on hard disk
		with open('memorysaved.pkl', 'wb') as f:
			pickle.dump(self.memory, f)



		
class Pacman:
	def __init__(self):
		self.sample_batch_size=32  # Batch size used to train in replay memory. 
		self.episodes= 20  # Number of trials.
		self.steps=0     # Will keep track of number of timesteps played till now.
		self.env=gym.make('MsPacman-ram-v0')
		self.state_size=self.env.observation_space.shape[0]
		#print(self.state_size)
		self.action_size=self.env.action_space.n
		self.player=Player(self.state_size, self.action_size)
		

	
	def learn(self):

		try:
			max_score=0
			sum_scores=0  # Which will keep sum of scores achieved in episodes

			rewards=[]
			#testing_scores=[]
			mean_rewards=[]
			

			for episode in range(self.episodes):

				# I will use a frame skip of size 4
				zero_steps=0   # No of steps with reward 0 in this episode, add it to episode reward
				frame_counter=0

				state=self.env.reset()
				#print(state.shape)
				state=state/256.0
				#print(state.shape)
				state=np.reshape(state, [1, self.state_size])

				done=False
				
				episode_reward=0   # Total reward in an episode.
				cur_lives=3        # Pacman starts with 3 lives.
				
				while not done:
					self.env.render()

					if frame_counter%6==0:					
						action=self.player.act(state)

					print("Action: {}".format(action))
					next_state,reward,done,info=self.env.step(action)
					
					next_state=next_state/256.0
					next_state=np.reshape(next_state, [1, self.state_size])

					self.steps+=1  # Time steps increased. 
	

					if self.steps%2000==0:  # For every 2000 steps target network is made equal to learning network
						self.player.target_train()


					if info['ale.lives']<cur_lives: # Someone made pacman dead
						reward=-100                   # Get immediate reward of -100 if pacman is dead
						cur_lives=info['ale.lives']  # Decrease cur_lives
						self.player.remember(state, action, reward, next_state, done)
						#Since pacman is dead its important for our neural net to fit on it.
						#self.player.replay(self.sample_batch_size,(state, action, reward, next_state, done))
					else:
						if reward==0:  # So that pacman tries to finish quicker
							reward=-2
							zero_steps+=1
						self.player.remember(state, action, reward, next_state, done)
						#if reward>=50:
						#	self.player.replay(self.sample_batch_size,(state, action, reward, next_state, done))
						#else:
					if self.steps%5==0:
						self.player.replay(self.sample_batch_size)

					print("Reward: {} done : {} info: {}".format(reward,done,info))

					"""	
					if done:  # If pacman exhausts all its lives is necessary to remember this. 
						self.player.remember(state, action, reward, next_state, done)
						self.player.replay(self.sample_batch_size)

					"""

					episode_reward+=reward

					state=next_state
					frame_counter+=1
					

				episode_reward+=2*zero_steps+300
				rewards.append(episode_reward)

				sum_scores+=episode_reward

				mean_rewards.append(sum_scores*1.0/(episode+1))

				#mean_rewards.append(np.mean(np.array(rewards)))
				#median_rewards.append(np.median(np.array(rewards)))

				print("Episode {}# Score: {} Exploration Rate: {} Mean reward: {}".format(episode+1,episode_reward,self.player.exploration_rate,mean_rewards[-1]))
				print("Number of timesteps played till now: {}".format(self.steps))

				if episode_reward>=max_score:
					#print("Got a high score.")
					max_score=episode_reward

				if episode%50==0:
					self.player.save()

				if episode%100==0:
					self.player.save_memory()

				
				if episode%500==0:
					with open('meanscores.pkl', 'wb') as f:
						pickle.dump(mean_rewards, f)

				

				"""

				#print("Now I am going to test my model on testing state...")
				# After each episode I will test my model on testing states.
				#avg_score=0   # Avg of maximum cumulative scores I can get on my testing states
				for test_state in self.player.testing_states:
					act_values=self.player.model.predict(test_state)
					avg_score+=np.amax(act_values[0])
				avg_score=avg_score/len(self.player.testing_states)
				print("Average score of all testing states is {}".format(avg_score))
				testing_scores.append(avg_score)
				"""

			print("Highest score obtained during training is: {}".format(max_score))
			print("Mean score over all episodes is: {}".format(mean_rewards[-1]))
			

		finally:
			self.player.save()
			self.player.save_memory()
			self.env.render(close=True)
			
	def play(self):

		rewards=[]

		for episode in range(20):

			state=self.env.reset()
			state=state/256.0
			state=np.reshape(state,[1,self.state_size])
			done=False
			step=0

			episode_reward=0   # Total reward in an episode

			while not done:
				rgb = self.env.render('rgb_array')
				upscaled=repeat_upsample(rgb,4, 4)
				viewer.imshow(upscaled)
				#self.env.render()
				action=self.player.act(state)
				next_state,reward,done,_=self.env.step(action)
				next_state=next_state/256.0
				episode_reward+=reward
				next_state=np.reshape(next_state,[1,self.state_size])
				state=next_state
				step+=1

			print("Episode {}# Score: {}".format(episode,episode_reward))
				rewards.append(episode_reward)

		self.env.render(close=True)


		print("Average score over {} episodes is {}".format(20,np.mean(np.array(rewards))))



if __name__ == "__main__":
	pacman = Pacman()
	#pacman.prepare()
	pacman.play()
