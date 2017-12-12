import random
import os
import operator
import gym
from skimage import io, color, transform
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt

class Agent:
        #
    def __init__(self,action_size,epsilon=1.0,experience_replay_capacity=1000,minibatch_size=32,learning_rate=0.1,gamma=0.95,preprocess_image_dim=84):
        self.action_size = action_size
        self.epsilon = epsilon
        self.experience_replay_capacity = experience_replay_capacity
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.preprocess_image_dim = preprocess_image_dim

        self.memory = []
        self.ere_counter = 0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.92
        self.model = self.create_model()


    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3,3),input_shape=(1,84,84), strides=(1,1), padding='same', data_format='channels_first', activation='relu', use_bias=True, bias_initializer='zeros'))
        model.add(Conv2D(32, (3,3),input_shape=(1,84,84), strides=(1,1), padding='same', data_format='channels_first', activation='relu', use_bias=True, bias_initializer='zeros'))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None))

        #model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', data_format=None, activation='relu', use_bias=True, bias_initializer='zeros'))
        model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', data_format=None, activation='relu', use_bias=True, bias_initializer='zeros'))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None))

        model.add(Flatten())

        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='tanh'))
        model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = self.learning_rate))

        return model


    def append_experience_replay_example(self, experience_replay_example):
        """
        Add an experience replay example to our agent's replay memory. If
        memory is full, overwrite previous examples, starting with the oldest
        """
        if (self.ere_counter >= self.experience_replay_capacity):
            self.ere_counter = 0
            self.memory[self.ere_counter] = experience_replay_example
        else:
            self.memory.append(experience_replay_example)
        self.ere_counter += 1


    def preprocess_observation(self, observation, prediction=False):
        """
        Helper function for preprocessing an observation for consumption by our
        deep learning network
        """
        #print(observation.shape)
        grayscale_observation = color.rgb2gray(observation)
        #print(grayscale_observation.shape) (210,160)
        resized_observation = transform.resize(grayscale_observation, (1, self.preprocess_image_dim, self.preprocess_image_dim)).astype('float32')
        if prediction:
            resized_observation = np.expand_dims(resized_observation, 0)
        #print(resized_observation.shape) (1,84,84)
        return resized_observation

    def take_action(self, observation):
        """
        Given an observation, the model attempts to take an action
        according to its q-function approximation
        """
        observation = np.array(observation)
        observation = np.reshape(observation, [1,1,self.preprocess_image_dim,self.preprocess_image_dim])

        #print(observation.shape) (1,84,84)
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
        minibatch = random.sample(self.memory, self.minibatch_size)
        for obs, action, reward, next_obs, done in minibatch:
            obs = np.reshape(np.array(obs),[1,1,self.preprocess_image_dim,self.preprocess_image_dim])
            next_obs = np.reshape(np.array(next_obs),[1,1,self.preprocess_image_dim,self.preprocess_image_dim])
            target = reward
            if not done:
                target = reward + self.gamma*np.amax(self.model.predict(next_obs)[0])
            target_f = self.model.predict(obs)
            target_f[0][action] = target
            self.model.fit(obs, target_f, epochs=1, verbose=0)
        if (self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay

    def shortLearn(self, shortMemory):
        """
        Allow the model to collect examples from its experience replay memory
        and learn from them
        """
        minibatch = shortMemory
        for obs, action, reward, next_obs, done in minibatch:
            obs = np.reshape(np.array(obs),[1,1,self.preprocess_image_dim,self.preprocess_image_dim])
            next_obs = np.reshape(np.array(next_obs),[1,1,self.preprocess_image_dim,self.preprocess_image_dim])
            target = reward
            if not done:
                target = reward + self.gamma*np.amax(self.model.predict(next_obs)[0])
            target_f = self.model.predict(obs)
            target_f[0][action] = target
            self.model.fit(obs, target_f, epochs=1, verbose=0)
    
#####
# Hyperparameters
#####

GAME_TYPE = 'MsPacman-v0'

#environment parameters
NUM_EPISODES = 50
MAX_TIMESTEPS = 5
FRAME_SKIP = 2
PHI_LENGTH = 4

#agent parameters
NAIVE_RANDOM = False
EPSILON = 1.0
GAMMA = 0.95
EXPERIENCE_REPLAY_CAPACITY = 1000
MINIBATCH_SIZE = 70
LEARNING_RATE = 0.2
PREPROCESS_IMAGE_DIM = 84
SCORE_LIST = []


def run_simulation():
    """
    Entry-point for running Ms. Pac-man simulation
    """

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
    print ("Experience Replay Capacity: %s" % EXPERIENCE_REPLAY_CAPACITY)
    print ("Minibatch Size: %s" % MINIBATCH_SIZE)
    print ("Learning Rate: %s" % LEARNING_RATE)

    #initialize agent
    agent = Agent(action_size = ACTION_SIZE,epsilon=EPSILON,
                experience_replay_capacity=EXPERIENCE_REPLAY_CAPACITY,
                minibatch_size=MINIBATCH_SIZE,
                learning_rate=LEARNING_RATE,gamma = GAMMA,preprocess_image_dim=PREPROCESS_IMAGE_DIM)

    #initialize auxiliary data structures
    # S_LIST = [] # Stores PHI_LENGTH frames at a time
    # TOT_FRAMES = 0  # Counter of frames covered till now

    for i_episode in range(NUM_EPISODES):
        obs = ENV.reset()
        EPISODE_REWARD = 0
        time = 0
        obsProc = agent.preprocess_observation(obs)
        step = 1
        REWARD = 0
        oldREWARD=0
        shortMemory = []
        EPISODE_PERFORMANCE = 0 
        start=False
        while True:
            ENV.render()
            #OBS = agent.preprocess_observation(OBS)
            #ensure that S_LIST is populated with PHI_LENGTH frames
            """
            if TOT_FRAMES < PHI_LENGTH:
                S_LIST.append(agent.preprocess_observation(OBS))
                TOT_FRAMES += 1
                continue
            """
            #X = np.array(S_LIST)
            #print(X.shape) #(4,1,84,84)

            # call take_action
            ACTION = agent.take_action(obsProc)
            #print(ACTION)

            newObs, newREWARD, DONE, INFO = ENV.step(ACTION) # NEXT_OBS is a numpy.ndarray of shape(210,160,3)
            # LIVES = LIVES.get('ale.lives')
            # Calculation of Reward
            EPISODE_REWARD += newREWARD

            if (newREWARD!=0):
                start = True
                if oldREWARD <0:
                    REWARD = 0
                    oldREWARD = 0
                    shortMemory = []
                oldREWARD += 1
                REWARD = 1
                step = 1
            else:
                REWARD = 0
                step +=1 
                if (step%8 == 0):
                    if oldREWARD >0 :
                        shortMemory = []
                        REWARD = 0
                        oldREWARD = 0 
                    oldREWARD += -1
                    REWARD = -1
            EPISODE_PERFORMANCE += REWARD          
            '''
            if (1):
                print(REWARD)
            '''

            #print(NEXT_OBS, REWARD, DONE,'\n\n\n\n\n\n\n\n')
            newObsProc = agent.preprocess_observation(newObs) # shape(1,84,84)

            EREG = [obsProc, ACTION, REWARD, newObsProc, DONE]
            #print(EREG)

            agent.append_experience_replay_example(EREG)
            shortMemory.append(EREG)

            obsProc = newObsProc

            if (((oldREWARD > 2 ) & (REWARD > 0)) & start):   
                print(str(REWARD) + " + " + str(len(shortMemory)))
                oldREWARD = 0
                print("shortLearn")
                for i in range(len(shortMemory)):
                    shortMemory[i][2] = REWARD
                    agent.append_experience_replay_example(shortMemory[i])
                agent.shortLearn(shortMemory)

            if DONE:
                print("DEADLearn " + str(len(shortMemory)))
                dead= []
                for i in range(min(45,len(shortMemory))):
                    er = shortMemory.pop()
                    er[2]= -100
                    dead.append(er)
                agent.shortLearn(dead)
                print("episode:{}/{}, score: {}, performance {}, e = {}".format(i_episode, NUM_EPISODES, EPISODE_REWARD,EPISODE_PERFORMANCE, agent.epsilon))
                break

            #update state list with next observation
            """
            S_LIST.append(agent.preprocess_observation(OBS))
            S_LIST.pop(0)
            """
            
            time += 1


        SCORE_LIST.append(EPISODE_REWARD)
        
        if (len(agent.memory)>agent.minibatch_size):
            agent.learn()

def plot_rewards(score_list, episode_num):
    thefile = open('pacman3.txt', 'w')
    for item in score_list:
        thefile.write("%s\n" % item)
    thefile.close()

run_simulation()
plot_rewards(SCORE_LIST, NUM_EPISODES)