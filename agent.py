from enum import Enum
from os import name
import torch
import random
import numpy as np
from game import SnakeGameRL, Direction, Point, BLOCK_SIZE
from collections import deque
from model import QNet, QNetTrainer
from helper import plot
import pygame
import sys

MAX_MEMORY = 100000
BATCH_SITE = 1000
LR = 0.001

class Agent:
    def __init__(self, eps_subfactor=80, random_max=200):
        self.eps_factor = eps_subfactor
        self.random_max = random_max
        self.nb_games = 0
        self.epsilon = 0 # Parameter to control the randomness
        self.gamma = 0 # Discount rate (usually around 0.9 or 0.8)
        self.memory = deque(maxlen=MAX_MEMORY) # if exceeded, elements from left are dropped automatically
        self.model = QNet(11,256,3)
        self.trainer = QNetTrainer(self.model, lr=LR, gamma=self.gamma)
        # TODO: model, trainer

    def get_state(self, game:SnakeGameRL):
        """Get the current state of the game

        Args:
            game (SnakeGameRL): current game

        Returns:
            np.ndarray: state of the game
        """
        head = game.snake[0]
        p_l = Point(head.x-BLOCK_SIZE, head.y)
        p_r = Point(head.x+BLOCK_SIZE, head.y)
        p_u = Point(head.x, head.y-BLOCK_SIZE)
        p_d = Point(head.x, head.y+BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        dgr_straight = ((dir_r and game._close_collision(p_r)) or 
            (dir_l and game._close_collision(p_l)) or
            (dir_u and game._close_collision(p_u)) or
            (dir_d and game._close_collision(p_d)))

        dgr_right = ((dir_r and game._close_collision(p_d)) or
                    (dir_l and game._close_collision(p_u)) or
                    (dir_u and game._close_collision(p_r)) or 
                    (dir_d and game._close_collision(p_l))) 

        dgr_left = ((dir_r and game._close_collision(p_u)) or
                    (dir_l and game._close_collision(p_d)) or
                    (dir_u and game._close_collision(p_l)) or 
                    (dir_d and game._close_collision(p_r)))

        food_l = game.food.x < game.head.x
        food_r = game.food.x > game.head.x
        food_u = game.food.y < game.head.y
        food_d = game.food.y > game.head.y

        state = [
            dgr_straight,
            dgr_right,
            dgr_left,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            food_l,
            food_r,
            food_u,
            food_d
        ] 

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, finish):
        """Add the parameters of one step to memory

        Args:
            state ([type]): current state
            action ([type]): current action
            reward ([type]): current reward
            next_state ([type]): next_state
            finish ([type]): if game finished
        """
        self.memory.append((state, action, reward, next_state, finish)) # append as 1 tuple

    def train_long_memory(self):
        """Train at the end of each game
        """
        if len(self.memory) > BATCH_SITE:
            rdm_sample = random.sample(self.memory, BATCH_SITE) # A batch size list of tuples
        else:
            rdm_sample = self.memory

        states, actions, rewards, next_states, finishs = zip(*rdm_sample)
        self.trainer.train_step(states, actions, rewards, next_states, finishs) 

    def train_short_memory(self, state, action, reward, next_state, finish):
        """Train only for every game step

        Args:
            state ([type]): current state of the snake
            action ([type]): action used
            reward ([type]): reward value
            next_state ([type]): next state 
            finish ([type]): if game has finished
        """
        self.trainer.train_step(state, action, reward, next_state, finish)

    def get_action(self, state, train=True):
        """Get the action to perform. Tradeoff between exploration and exploitation

        Args:
            state ([type]): state with which to predict the action
        """
        self.epsilon = self.eps_factor - self.nb_games
        move = [0,0,0]
        if train and random.randint(0,self.random_max) < self.epsilon:
            # Sample an action randomly
            move_idx = random.randint(0,2)
            move[move_idx] = 1
        else:

            # Predict action from state
            state_torch = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state_torch)
            
            # Take max of the vector
            move_idx = torch.argmax(prediction).item()
            move[move_idx] = 1
        
        return move

def train(train_model_name="trained_model"):
    """Train the agent
    """
    scores = []
    mean_scores=[]
    total_score=0
    best_score=0
    agent = Agent()
    game = SnakeGameRL()
    
    while True:
        # get current state
        curr_state = agent.get_state(game)

        # get move
        move = agent.get_action(curr_state)

        # move the snake
        reward, finish, score = game.play_step(move)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(curr_state, move, reward, new_state, finish)
        
        # remember
        agent.remember(curr_state, move, reward, new_state, finish)

        if finish:
            # train long memory
            game._reset()
            agent.nb_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                agent.model.save(name=train_model_name)
                            
            print('Game ',agent.nb_games, 'Score: ', score, 'Record: ', best_score)
            scores.append(score)
            total_score+=score
            mean_scores.append(total_score/agent.nb_games)
            plot(scores, mean_scores)

def eval(model_name="final_model"):
    game = SnakeGameRL()
    agent = Agent()
    agent.model.load(name=model_name)
    agent.model.eval()

    while True:
        state = agent.get_state(game)
        action = agent.get_action(state, train=False)
        _, finish, score = game.play_step(action)

        if finish:
            print("Your score is {}".format(score))
            break
    
    pygame.quit()

if __name__ == '__main__':
    eval_model_name = "final_model.pth"
    train_model_name = "train_model.pth"
    
    eval_mode = True
    if len(sys.argv) == 1:
        print("No argument given .. set to evaluate mode as default")
    else:
        if sys.argv[1] == "eval":
            print("Set to evaluate mode as default")            
        elif sys.argv[1] == "train":
            print("Set to train mode as default")
            eval_mode = False
        else:
            print("Argument not recognizable .. set to evaluate mode as default")

    if eval_mode:
        eval(eval_model_name)
    else:
        train(train_model_name)
