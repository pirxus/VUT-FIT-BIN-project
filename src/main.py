"""@package main

Author: Simon Sedlacek
Email: xsedla1h@stud.fit.vutbr.cz

This is the main module of the program. Please, see the README.md file for
instructions.

"""

import pickle
import sys
import itertools
import gym
import ray
import argparse as ap
from config import *

from agent import Agent, AgentPool

@ray.remote
def eval_agent_pool(task, agent_pool):
    """Ray remote function, evaulating its supplied agent pool"""

    if task == 'cartpole': env = gym.make("CartPole-v1") # works, 4 4 2 2
    elif task == 'acrobot': env = gym.make("Acrobot-v1")
    elif task == 'ant': env = gym.make("BipedalWalker-v3")
    else: env = gym.make("LunarLander-v2")

    for i, agent in enumerate(agent_pool):
        observation = env.reset()

        total_reward = 0
        num_iterations = 1

        for k in range(num_iterations):

            for _ in range(MAX_ITER):

                p = True if task == 'ant' else False
                action = agent.forward(observation, p=p)

                observation, reward, done, info = env.step(action)
                total_reward += reward

                if done:
                    break

        # average the reward
        agent.fitness = total_reward / num_iterations

    return agent_pool

def train(task='cartpole', save_dir='./agents', n_proc=4, mutation_rate=0.05, crossover='uniform',
        population_size=200, max_generations=1000, crossover_rate=0.6):
    """ Main program branch for training the population """

    # init ray due to some problems at metacentrum
    ray.init(_temp_dir='/tmp/ray/')

    # create a pool of agents
    if task == 'cartpole':
        env = gym.make("CartPole-v1") # works, 4 4 2 2
        net_tuple = CARTPOLE_NET
    elif task == 'acrobot':
        env = gym.make("Acrobot-v1")
        net_tuple = ACROBOT_NET
    elif task == 'ant':
        env = gym.make("BipedalWalker-v3")
        net_tuple = ANT_NET
    elif task == 'lander':
        env = gym.make("LunarLander-v2")
        net_tuple = LANDER_NET
    else:
        print(f"Error, '{task}' is not a valid task specification. Exitting...")
        sys.exit(1)

    agent_pool = AgentPool(population_size, net_tuple, task, save_dir)

    for gen in range(max_generations):
        n = population_size // n_proc
        pool_of_pools = [ agent_pool.pool[i:i + n] for i in range(0, population_size, n) ]

        # send out the split up agent pool to the remote workers
        ret_pools = ray.get([ eval_agent_pool.remote(task, pool) for pool in pool_of_pools ])
        agent_pool.pool = list(itertools.chain.from_iterable(ret_pools))

        # once all the agents are evaluated, create the new generation
        agent_pool.evolve(mutation_rate=mutation_rate, print_stats=True,
                crossover_type=crossover, crossover_rate=crossover_rate)

    env.close()

def replay(agent_file):
    with open(agent_file, 'rb') as f:
        data = pickle.load(file=f)

    agent = Agent(net_tuple=data['net_shape'])
    agent.update_weights(data['weights'])
    task = data['task']

    if task == 'cartpole':
        env = gym.make("CartPole-v1") # works, 4 4 2 2
    elif task == 'acrobot':
        env = gym.make("Acrobot-v1")
    elif task == 'ant':
        env = gym.make("BipedalWalker-v3")
    else:
        env = gym.make("LunarLander-v2")


    total_reward = 0

    observation = env.reset()
    for _ in range(MAX_ITER):
        env.render()

        p = True if task == 'ant' else False
        action = agent.forward(observation, p=p)
        observation, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    env.close()

if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Genetic Algorithms for solving openAI gym environments.")
    parser.add_argument('--mode', type=str, required=False, default='train', help='Specify the program runtime mode: {train, replay}.')
    parser.add_argument('--task', type=str, required=False, default='cartpole', help='Specify the training task: {cartpole, ant, lander}.')
    parser.add_argument('--agent_path', type=str, required=False, default=None, help='Specify the path to the agnent file for visualization.')
    parser.add_argument('--save_dir', type=str, required=False, default='./agents/', help='Specify the destination directory for saving the best agents.')
    parser.add_argument('--mutation_rate', type=float, required=False, default=0.05, help='Specify the mutation rate.')
    parser.add_argument('--crossover', type=str, required=False, default='uniform', help='Specify the crossover type.')
    parser.add_argument('--crossover_rate', type=float, required=False, default=0.6, help='Specify the crossover rate for uniform crossover.')
    parser.add_argument('--population_size', type=int, required=False, default=200, help='Specify the population size.')
    parser.add_argument('--max_generations', type=int, required=False, default=1000, help='Specify the maximum number of generations.')
    parser.add_argument('--np', type=int, required=False, default=4, help='Specify the number of processes for parallelization.')
    args = parser.parse_args()

    # print the information about the experiment
    print(args)

    if args.mode == 'train':
        train(task=args.task, save_dir=args.save_dir, n_proc=args.np,
                mutation_rate=args.mutation_rate, crossover=args.crossover,
                population_size=args.population_size, max_generations=args.max_generations, crossover_rate=args.crossover_rate)
    elif args.mode == 'replay':
        replay(args.agent_path)
    else:
        print(f"Error, '{args.mode}' is not a valid runtime mode. Exitting...")
        sys.exit(1)

