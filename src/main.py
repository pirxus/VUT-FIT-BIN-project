import torch
import itertools
import numpy as np
import gym
import multiprocessing as mp
from copy import deepcopy
import ray

from agent import Agent, AgentPool

POPULATION_SIZE = 100
MP = True
PROC = 4
#task = 'cartpole'
task = 'ant'

# create a pool of agents
if task == 'cartpole':
    env = gym.make("CartPole-v1") # works, 4 4 2 2
    net_tuple = (4, 4, 2, 2)
elif task == 'acrobot':
    env = gym.make("Acrobot-v1")
    net_tuple = (6, 6, 4, 3)

elif task == 'ant':
    env = gym.make("BipedalWalker-v3")
    net_tuple = (24, 16, 12, 4)
else:
    env = gym.make("LunarLander-v2")
    net_tuple = (8, 8, 6, 4)

agent_pool = AgentPool(POPULATION_SIZE, net_tuple)


@ray.remote
def eval_agent_pool(agent_pool):
    if task == 'cartpole': env = gym.make("CartPole-v1") # works, 4 4 2 2
    elif task == 'acrobot': env = gym.make("Acrobot-v1")
    elif task == 'ant': env = gym.make("BipedalWalker-v3")
    else: env = gym.make("LunarLander-v2")

    for i, agent in enumerate(agent_pool):
        observation = env.reset()

        total_reward = 0
        num_iterations = 1

        for k in range(num_iterations):

            for _ in range(500):

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


for gen in range(1000):
    if MP:
        n = POPULATION_SIZE // PROC
        pool_of_pools = [ agent_pool.pool[i:i + n] for i in range(0, POPULATION_SIZE, n) ]
        ret_pools = ray.get([ eval_agent_pool.remote(pool) for pool in pool_of_pools ])
        agent_pool.pool = list(itertools.chain.from_iterable(ret_pools))
        agent_pool.evolve(mutation_rate=0.1, print_stats=True)

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
