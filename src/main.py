import torch
import pickle
import sys
import itertools
import numpy as np
import gym
import multiprocessing as mp
from copy import deepcopy
import ray
import argparse as ap

from agent import Agent, AgentPool

POPULATION_SIZE = 200
MP = True
PROC = 4
#task = 'cartpole'
task = 'ant'


# TODO: replay

@ray.remote
def eval_agent_pool(task, agent_pool):
    if task == 'cartpole': env = gym.make("CartPole-v1") # works, 4 4 2 2
    elif task == 'acrobot': env = gym.make("Acrobot-v1")
    elif task == 'ant': env = gym.make("BipedalWalker-v3")
    else: env = gym.make("LunarLander-v2")

    for i, agent in enumerate(agent_pool):
        observation = env.reset()

        total_reward = 0
        num_iterations = 1

        for k in range(num_iterations):

            for _ in range(700):

                p = True if task == 'ant' else False
                action = agent.forward(observation, p=p)
                if i == 0:
                    pass
                #action = np.random.choice(range(4), 1, p=pred).item()
                observation, reward, done, info = env.step(action)
                total_reward += reward ** 3

                if done:
                    break

        # average the reward
        agent.fitness = total_reward / num_iterations

    return agent_pool

def train(task, save_dir):

    # create a pool of agents
    if task == 'cartpole':
        env = gym.make("CartPole-v1") # works, 4 4 2 2
        net_tuple = (4, 4, 2, 2)
    elif task == 'acrobot':
        env = gym.make("Acrobot-v1")
        net_tuple = (6, 6, 4, 3)

    elif task == 'ant':
        env = gym.make("BipedalWalker-v3")
        net_tuple = (24, 24, 16, 4)
    elif task == 'lander':
        env = gym.make("LunarLander-v2")
        net_tuple = (8, 36, 24, 4)
    else:
        print(f"Error, '{task}' is not a valid task specification. Exitting...")
        sys.exit(1)

    agent_pool = AgentPool(POPULATION_SIZE, net_tuple, task, save_dir)


    for gen in range(2000):
        if MP:
            n = POPULATION_SIZE // PROC
            pool_of_pools = [ agent_pool.pool[i:i + n] for i in range(0, POPULATION_SIZE, n) ]
            ret_pools = ray.get([ eval_agent_pool.remote(task, pool) for pool in pool_of_pools ])
            agent_pool.pool = list(itertools.chain.from_iterable(ret_pools))
            agent_pool.evolve(mutation_rate=0.05, print_stats=True, crossover_type='uniform')

        else:
            for i, agent in enumerate(agent_pool.pool):
                observation = env.reset()

                total_reward = 0
                num_iterations = 1

                for k in range(num_iterations):

                    for j in range(500):
                        if i == 0 and k == 0 and (gen + 1) % 10 == 0:
                            env.render()
                            pass

                        #action = env.action_space.sample() # your agent here (this takes random actions)
                        pred = agent.forward(observation, p=True)
                        if i == 0:
                            pass
                        #action = np.random.choice(range(4), 1, p=pred).item()
                        action = pred
                        observation, reward, done, info = env.step(action)

                        """
                        if reward == 200:
                            total_reward += 2 * reward
                        else:
                            total_reward += reward
                            """
                        total_reward += reward

                        if done:
                            break

                # average the reward
                agent.fitness = total_reward / num_iterations


            # select the parents
            agent_pool.evolve(mutation_rate=0.1, print_stats=True)

            # generate the new population
            # mutation
            # again!!!

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
    for _ in range(700):
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
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.task, args.save_dir)
    elif args.mode == 'replay':
        replay(args.agent_path)
    else:
        print(f"Error, '{args.mode}' is not a valid runtime mode. Exitting...")
        sys.exit(1)

