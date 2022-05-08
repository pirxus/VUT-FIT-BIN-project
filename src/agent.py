"""@package agent

Author: Simon Sedlacek
Email: xsedla1h@stud.fit.vutbr.cz

In this module, the classes for Agent and AgentPool are implemented.
"""

import numpy as np
import os
import pickle
from net import NpNet, softmax
from config import *

class Agent:
    """ This class represents one agent in the population """
    def __init__(self, agent_id=0, net_tuple=None) -> None:
        if net_tuple is not None:
            self.net = NpNet(*net_tuple)
        else:
            self.net = NpNet()
        self.fitness: float = 0
        self.fitness_prev: float = 0
        self.agent_id = agent_id

    def backup_and_reset_fitness(self):
        self.fitness_prev = self.fitness
        self.fitness = 0

    def reset_agent(self):
        self.fitness_prev = 0
        self.fitness = 0

    @staticmethod
    def create_agent_pool(num_agents, net_tuple=None):
        return [ Agent(k, net_tuple) for k in range(num_agents) ]

    def forward(self, x, p=False):
        prediction = self.net.forward(x)
        if not p:
            return int(np.argmax(softmax(prediction)))
        return prediction


    def serialize_weights(self):

        l1 = np.concatenate((np.ravel(self.net.w1), self.net.b1))
        l2 = np.concatenate((np.ravel(self.net.w2), self.net.b2))
        l3 = np.concatenate((np.ravel(self.net.w3), self.net.b3))

        return np.concatenate((l1, l2, l3))

    def update_weights(self, params: np.ndarray):
        """ Update the agent's network parameters from a numpy array

        The numpy array is a flattened array from which the weight
        and bias matrices are reconstructed.

        """

        d1 = self.net.input_dim * self.net.hidden_dim_1
        d2 = d1 + self.net.hidden_dim_1

        self.net.w1 = params[:d1].reshape((self.net.hidden_dim_1, self.net.input_dim))
        self.net.b1 = params[d1:d2]

        d1 = d2
        d2 = d1 + self.net.hidden_dim_1 * self.net.hidden_dim_2
        d3 = d2 + self.net.hidden_dim_2

        self.net.w2 = params[d1:d2].reshape((self.net.hidden_dim_2, self.net.hidden_dim_1))
        self.net.b2 = params[d2:d3]

        d1 = d3
        d2 = d1 + self.net.hidden_dim_2 * self.net.out_dim
        d3 = d2 + self.net.out_dim

        self.net.w3 = params[d1:d2].reshape((self.net.out_dim, self.net.hidden_dim_2))
        self.net.b3 = params[d2:d3]

    def mutate(self, p=0.05):
        """ Mutate the chromosomes of an agent

        Mutate the chromosomes of an agent. Each genome is altered
        with a small probability p and the delta used in the alteration
        is drawn from a standard normal distribution scaled by
        a factor of 0.2.

        """

        # get the serialized network parameters
        params = self.serialize_weights()

        # get a boolean array for indices where the mutation will be performed
        perform_mutation = np.random.random(params.size) < p

        # get the mutation values and update the params
        delta = np.random.normal(scale=0.2, size=params.size) * perform_mutation
        params += delta
        params = np.clip(params, -2, 2)

        self.update_weights(params)

class AgentPool:
    def __init__(self, num_agents, net_tuple=None, task='cartpole', save_dir='./agents/') -> None:
        self.pool = Agent.create_agent_pool(num_agents, net_tuple)
        self.population_size = num_agents
        self.best_fitness: float = 0
        self.generation = 0
        self.net_tuple = net_tuple
        self.next_id = num_agents
        self.task = task
        self.save_dir = save_dir

    def evolve(self, mutation_rate=0.05, print_stats=False, crossover_type='default',
            crossover_rate=0.6):
        """ Produce a new generation of agents

        This method wraps up the current generataion's simulation,
        saves some statistics about the generation and creates
        a new generation of agents.

        """

        self.generation += 1

        # sort the agents by fitness
        self.pool.sort(key=lambda x: x.fitness, reverse=True)

        if print_stats:
            print(f"=========== generation {self.generation} summary =============")

            print("Printing top five best agents")

            for agent in self.pool[:5]:
                print(f"ID: {agent.agent_id},\tfitness: {agent.fitness:.3f},\t")

            fitness_array = np.ndarray((len(self.pool),))
            for i in range(fitness_array.size):
                fitness_array[i] = self.pool[i].fitness

            print("\nMean fitness:", round(np.mean(fitness_array), 3), "\n")
            print("\nMedian fitness:", round(np.median(fitness_array), 3), "\n")


        new_population = []
        new_population.append(self.pool[0]) # use the elite

        if self.population_size > 50:
            new_population.append(self.pool[1])

        # select a random number of parents
        num_parents = np.random.randint(self.population_size // 10, self.population_size // 5) if self.population_size > 50 else self.population_size

        parents = self.pool[:num_parents]
        while len(new_population) < self.population_size:

            p1, p2 = self.roulette_selection(parents, selection_size=2)
            if crossover_type == 'separate':
                ch1, ch2 = self.crossover_separate(p1, p2)
            elif crossover_type == 'uniform':
                ch1, ch2 = self.crossover_uniform(p1, p2, p=crossover_rate)
            elif crossover_type == 'blx':
                ch1, ch2 = self.crossover_blxa(p1, p2, alpha=0.5)
            else:
                ch1, ch2 = self.crossover(p1, p2, points=1)

            ch1.mutate(p=mutation_rate)
            ch2.mutate(p=mutation_rate)

            new_population.extend([ch1, ch2])

        new_population.sort(key=lambda x: x.fitness, reverse=True)
        self.backup_best_agent(output_dir=self.save_dir, gen_interval=BCKP_INTERVAL)
        self.generate_report(output_dir=self.save_dir)

        for individual in new_population:
            individual.backup_and_reset_fitness()
        self.pool = new_population[:self.population_size]


    def roulette_selection(self, agent_pool: list, selection_size=2):
        """ Perform roulette agent selection

        Create a list of parents intended for crossover utilizing the
        roulette selection principle.

        """

        roulette = sum([ candidate.fitness for candidate in agent_pool ])
        sel_prob = [ candidate.fitness / roulette for candidate in agent_pool ]
        selected = np.random.choice(agent_pool, size=(selection_size, ), replace=False, p=sel_prob)
        return selected


    def crossover_uniform(self, p1: Agent, p2: Agent, p=0.8):
        ch1 = p1.serialize_weights()
        ch2 = p2.serialize_weights()

        cross = np.random.random(ch1.size) < p

        out1 = ~cross * ch1 + cross * ch2
        out2 = ~cross * ch2 + cross * ch1

        # create new agents
        a1 = Agent(self.next_id, net_tuple=self.net_tuple)
        a2 = Agent(self.next_id + 1, net_tuple=self.net_tuple)
        self.next_id += 2

        # set the network parameters for each agent
        a1.update_weights(out1)
        a2.update_weights(out2)

        return a1, a2

    def crossover_blxa(self, p1: Agent, p2: Agent, alpha=0.5):
        ch1 = p1.serialize_weights()
        ch2 = p2.serialize_weights()

        dist = np.abs(np.subtract(ch1, ch2))
        out1 = np.random.uniform(low=np.minimum(ch1, ch2) - (alpha * dist),
                high=np.maximum(ch1, ch2) + (alpha * dist), size=ch1.size)
        out2 = np.random.uniform(low=np.minimum(ch1, ch2) - (alpha * dist),
                high=np.maximum(ch1, ch2) + (alpha * dist), size=ch1.size)


        # create new agents
        a1 = Agent(self.next_id, net_tuple=self.net_tuple)
        a2 = Agent(self.next_id + 1, net_tuple=self.net_tuple)
        self.next_id += 2

        # set the network parameters for each agent
        a1.update_weights(out1)
        a2.update_weights(out2)

        return a1, a2

    def crossover_separate(self, p1: Agent, p2: Agent):
        """ Perform parent crossover and produce offsprings

        Performs crossover for two parents and produces
        two new descendants. The crossover performed on
        each array of weights and biases separately in the
        one-point manner.

        """

        w11 = np.ravel(p1.net.w1)
        w12 = np.ravel(p2.net.w1)
        b11 = p1.net.b1
        b12 = p2.net.b1
        w21 = np.ravel(p1.net.w2)
        w22 = np.ravel(p2.net.w2)
        b21 = p1.net.b2
        b22 = p2.net.b2
        w31 = np.ravel(p1.net.w3)
        w32 = np.ravel(p2.net.w3)
        b31 = p1.net.b3
        b32 = p2.net.b3
        out1 = []
        out2 = []

        parents = [(w11, w12), (b11, b12), (w21, w22), (b21, b22), (w31, w32), (b31, b32)]
        for (ch1, ch2) in parents:
            point = np.random.randint(1, ch1.size)
            o1 = np.copy(ch1)
            o2 = np.copy(ch2)
            o1[point:] = ch2[point:]
            o2[point:] = ch1[point:]
            out1.append(o1)
            out2.append(o2)

        # create new agents
        """
        a1 = Agent(self.last_id + 1)
        a2 = Agent(self.last_id + 2)
        self.last_id += 2
        """
        a1 = Agent(self.next_id, net_tuple=self.net_tuple)
        a2 = Agent(self.next_id + 1, net_tuple=self.net_tuple)
        self.next_id += 2

        out1 = np.concatenate(out1)
        out2 = np.concatenate(out2)

        # set the network parameters for each agent
        a1.update_weights(out1)
        a2.update_weights(out2)

        return a1, a2

    def crossover(self, p1: Agent, p2: Agent, points=1):
        """ Perform parent crossover and produce offsprings

        Performs crossover for two parents and produces
        two new descendants. The crossover is performed on
        a serialized array of all the weights and biases of
        the agent's neural network.

        """

        ch1 = p1.serialize_weights()
        ch2 = p2.serialize_weights()
        out1 = np.ndarray(ch1.size)
        out2 = np.ndarray(ch2.size)

        # generate the crossover points
        co_points = [ch1.size]
        for _ in range(points):
            point = ch1.size
            while point in co_points:
                point = np.random.randint(1, ch1.size)

            co_points.append(point)

        co_points.sort()

        flip = True
        prev = 0
        for point in co_points:
            if flip:
                out1[prev:point] = ch1[prev:point]
                out2[prev:point] = ch2[prev:point]
                flip = False
            else:
                out1[prev:point] = ch2[prev:point]
                out2[prev:point] = ch1[prev:point]
                flip = True

            prev = point

        # create new agents
        a1 = Agent(self.next_id, net_tuple=self.net_tuple)
        a2 = Agent(self.next_id + 1, net_tuple=self.net_tuple)
        self.next_id += 2

        # set the network parameters for each agent
        a1.update_weights(out1)
        a2.update_weights(out2)

        return a1, a2


    def generate_report(self, output_dir='./agents/', fname='stats.csv'):
        """ Write the stats about the current generation into a file """

        # if needed create the destination path for the file..
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, fname), 'a') as f:
            fitness = []
            for agent in self.pool:
                fitness.append(str(agent.fitness))

            f.write(';'.join(fitness) + "\n")



    def backup_best_agent(self, output_dir='./agents/', gen_interval=1):
        """ Backup the weights of the best snake in the current generation """

        if self.generation % gen_interval != 0:
            return

        # get the best performing agent
        best_agent = sorted(self.pool, key= lambda x: x.fitness, reverse=True)[0]

        # save the agent
        if output_dir:
            # if needed create the destination path for the file..
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # serialize the agent's weights and write the file..
            weights = {'id': best_agent.agent_id,
                    'weights': best_agent.serialize_weights(),
                    'fitness': best_agent.fitness,
                    'net_shape': self.net_tuple,
                    'task': self.task,
                    'gen': self.generation}
            with open(output_dir + '/' +
                    f"task:{self.task}_gen:{self.generation}_id:{best_agent.agent_id}_fit:{best_agent.fitness:.2f}.pkl",
                    'wb') as f:
                pickle.dump(weights, file=f)

