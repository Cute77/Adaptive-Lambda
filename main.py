from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import gym
import numpy as np
import matplotlib.pyplot as plt
import my_optim
from model import ActorCritic
from test import test
from train import train
from tensorboardX import SummaryWriter
import time


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda1', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 0.8)')
parser.add_argument('--gae-lambda2', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 0.8)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=1,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=200,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', action="store_true",
                    help='use an optimizer without shared momentum.')

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    env = gym.make('MountainCar-v0').unwrapped
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    num_done = mp.Value('i', 0)
    num_episode = mp.Value('i', 0)
    reward_sum = mp.Value('i', 0)
    arr = mp.Array('i', [])
    lock = mp.Lock()
    writer = SummaryWriter("logs/fig"+str(args.gae_lambda1)+"_"+ str(args.gae_lambda2), max_queue = 1)    

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter, num_done, num_episode, reward_sum, lock))
    p.start()
    processes.append(p)
    def write(a,b,c):
        i = 0
        while counter.value < 120000000:
            print(a.value, b.value, c.value, counter.value / 10000)
            writer.add_scalar("test/reward", a.value, counter.value / 10000)
            writer.add_scalar("train/rate", b.value * 1.0 / c.value, counter.value / 10000)
            i = i + 1
            time.sleep(10)
    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, num_done, num_episode, arr, lock, optimizer))
        p.start()
        processes.append(p)
    p = mp.Process(target = write, args=(reward_sum, num_done, num_episode))
    p.start()
    processes.append(p) 
    for p in processes:
        p.join()

    
    
