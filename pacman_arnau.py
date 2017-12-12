import random
import os
import operator
import gym
from skimage import io, color, transform
import numpy as np
import random

import keras
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from keras.optimizers import Adam

from collections import deque
import matplotlib.pyplot as plt

class Agent:
        #
    def __init__(self,action_size,epsilon=1.0,epsilon_min=0.01,epsilon_decay=0.99,memory_capacity=1000,minibatch_size=32,learning_rate=0.01,gamma=0.95,preprocess_image_dim=84):
        self.action_size = action_size
        self.epsilon = epsilon  # exploration rate
        self.memory_capacity = memory_capacity
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.preprocess_image_dim = preprocess_image_dim

        self.memory = []
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.create_model()


    def create_model(self):

        model = Sequential()
        model.add(Conv2D(16, (3,3), input_shape=(210,160,3), strides=(2,2), padding='same', data_format='channels_last', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None))

        model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', data_format=None, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None))

        model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', data_format=None, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None))

        model.add(Flatten())

        model.add(Dense(200, activation='relu'))
        model.add(Dense(200,activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate)) #categorical_crossentropy

        return model


    def update_memory(self, result):
        """
        Add an experience replay example to our agent's replay memory. If
        memory is full, overwrite previous examples, starting with the oldest
        """
        if (len(self.memory) >= self.memory_capacity):
            self.memory.pop(0)
            self.memory.append(result)
        else:
            self.memory.append(result)
        


    def preprocess_observation(self, observation):
        """
        Helper function for preprocessing an observation for consumption by our
        deep learning network.
        Enviroment observations are 210x160x3 8 bit images.
        """
        #grayscale_observation = color.rgb2gray(observation)
        observation_dim = observation.shape
        resized_observation = observation.reshape(1, observation_dim[0], observation_dim[1], observation_dim[2])
        resized_observation = resized_observation.astype('float32')
        resized_observation = resized_observation/255
        return resized_observation

    def take_action(self, observation):
        """
        Given an observation, the model attempts to take an action
        according to its q-function approximation.abs
        Possible actions are 9 in Pacman : 0,...,8
        0: no change, 1: up, 2: right, 3: left, 4: down, ...
        """
        if (np.random.rand() <= self.epsilon):
            action = random.randrange(self.action_size)
            return action
        act_values = self.model.predict(observation) # Forward Propagation
        action = np.argmax(act_values[0])
        return action

    def learn(self):
        """
        Allow the model to collect examples from its experience replay memory
        and learn from them
        """
        last_experience = self.memory[-1]
        Q_target = np.zeros(self.action_size)
        Q_target[last_experience[1]] = last_experience[2]
        self.model.fit(last_experience[0], Q_target[None,:], epochs=5, verbose=0)
        minibatch = random.sample(self.memory, self.minibatch_size)
        for obs, action, reward, next_obs, done in minibatch:
            Q_target = np.zeros(self.action_size)
            if done:
                Q_target[action] = reward
            if not done:
                Q_target[action] = reward + self.gamma*np.amax(self.model.predict(next_obs)[0])
            self.model.fit(obs, Q_target[None,:], epochs=1, verbose=0)
        if (self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay
    
    
        



def run_simulation(file):

    for aaa in range(10):
        #####
        # Hyperparameters
        #####

        GAME_TYPE = 'MsPacman-v0'

        #environment parameters
        NUM_EPISODES =100
        MAX_TIMESTEPS = 5
        FRAME_SKIP = 2
        PHI_LENGTH = 4

        #agent parameters
        NAIVE_RANDOM = False
        EPSILON = 1.0
        EPSILON_MIN=0.1
        EPSILON_DECAY=0.97
        GAMMA = 0.95
        memory_capacity = 1000
        MINIBATCH_SIZE = 50
        LEARNING_RATE = 0.3*(aaa+1)
        PREPROCESS_IMAGE_DIM = 84
        SCORE_LIST = []
        """
        Entry-point for running Ms. Pac-man simulation
        """
        
        #initialize enviroment
        ENV = gym.make(GAME_TYPE)
        ACTION_SIZE = ENV.action_space.n
        DONE = False

        #print game parameters
        print ("~~~Environment Parameters~~~")
        print ("Num episodes: %s" % NUM_EPISODES)
        print ("Max timesteps: %s" % MAX_TIMESTEPS)
        print ("Action space: %s" % ACTION_SIZE)
        print()
        print ("~~~Agent Parameters~~~")
        print ("Naive Random: %s" % NAIVE_RANDOM)
        print ("Epsilon: %s" % EPSILON)
        print ("Experience Replay Capacity: %s" % memory_capacity)
        print ("Minibatch Size: %s" % MINIBATCH_SIZE)
        print ("Learning Rate: %s" % LEARNING_RATE)

        #initialize agent
        agent = Agent(  action_size = ACTION_SIZE,
                        epsilon=EPSILON,
                        epsilon_min=EPSILON_MIN,
                        epsilon_decay=EPSILON_DECAY,
                        memory_capacity=memory_capacity,
                        minibatch_size=MINIBATCH_SIZE,
                        learning_rate=LEARNING_RATE,
                        gamma = GAMMA,
                        preprocess_image_dim=PREPROCESS_IMAGE_DIM)

        
        for i_episode in range(NUM_EPISODES):
            episode_reward = 0
            time = 0
            new_observation = ENV.reset()
            #Preprocess observation for correct format
            new_observation_preprocessed = agent.preprocess_observation(new_observation)
            
            
            while True:
                #For Rendering the enviroment at the beggining
                ENV.render()
                
                # Take action for that observation
                observation_preprocessed = new_observation_preprocessed
                action = agent.take_action(observation_preprocessed)
                
                # Perform action
                new_observation, reward, done, info = ENV.step(action)

                # Modify reward
                modified_reward=reward-1 #consideres spending time
                if done:
                    modified_reward = modified_reward-1000 # consideres end of game
                
                #Preprocess observation for correct format
                new_observation_preprocessed = agent.preprocess_observation(new_observation)
                
                # Update agent from the reward
                agent.update_memory([observation_preprocessed, action, modified_reward, new_observation_preprocessed, done])

                # The enviroment indicates the episode ended
                if done:
                    print("episode:{}/{}, score: {}, time: {}, e = {}".format(i_episode, NUM_EPISODES, episode_reward, time, agent.epsilon))
		    file.write("episode:{}/{}, score: {}, time: {}, e = {}".format(i_episode, NUM_EPISODES, episode_reward, time, agent.epsilon))
                    break
                
                #Update episode's reward and time
                episode_reward += reward
                time += 1

            
            if (i_episode%5==0):
                SCORE_LIST.append(episode_reward)
            if (len(agent.memory)>agent.minibatch_size):
               agent.learn()

def plot_rewards(score_list, episode_num):
    episode_num = [x for x in range(0,episode_num,5)]
    plt.plot(episode_num, score_list)
    plt.show()

# Main
file = open('results.txt', 'w')
run_simulation(file)
#plot_rewards(SCORE_LIST, NUM_EPISODES)
file.close()
