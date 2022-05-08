import pickle
import sys
import itertools
import numpy as np
import gym
from copy import deepcopy
import ray
import argparse as ap
from config import *

from agent import Agent, AgentPool

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

            for _ in range(MAX_ITER):

                p = True if task == 'ant' else False
                action = agent.forward(observation, p=p)
                if i == 0:
                    pass
                #action = np.random.choice(range(4), 1, p=pred).item()
                observation, reward, done, info = env.step(action)
                total_reward += reward

                if done:
                    break

        # average the reward
        agent.fitness = total_reward / num_iterations

    return agent_pool

def train(task, save_dir, n_proc, mutation_rate=0.05, crossover='uniform'):
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

    agent_pool = AgentPool(POPULATION_SIZE, net_tuple, task, save_dir)

    for gen in range(MAX_GENERATIONS):
        if MP:
            n = POPULATION_SIZE // n_proc
            pool_of_pools = [ agent_pool.pool[i:i + n] for i in range(0, POPULATION_SIZE, n) ]
            ret_pools = ray.get([ eval_agent_pool.remote(task, pool) for pool in pool_of_pools ])
            agent_pool.pool = list(itertools.chain.from_iterable(ret_pools))
            agent_pool.evolve(mutation_rate=mutation_rate,print_stats=True, crossover_type=crossover)

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
    parser.add_argument('--np', type=int, required=False, default=4, help='Specify the number of processes for parallelization.')
    args = parser.parse_args()

    # print the information about the experiment
    print(args)

    if args.mode == 'train':
        train(args.task, args.save_dir, args.np,
                mutation_rate=args.mutation_rate, crossover=args.crossover)
    elif args.mode == 'replay':
        replay(args.agent_path)
    else:
        print(f"Error, '{args.mode}' is not a valid runtime mode. Exitting...")
        sys.exit(1)

