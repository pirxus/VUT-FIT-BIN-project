# VUT-FIT-BIN-project
## Genetic algorithms for solving OpenAI Gym problesm

This application implements a simple framework for solving the problems available in the
OpenAI gym framework.

The genetic algorithm is used to optimize the weight and bias matrices
of the agent's neural network.


## Setup and usage

The application requires Python 3.

To setup the project and install the necessary dependencies,
simply run:
```
make install
```

To run the application in training mode, simply use the following
commmand:
```
make train
```

or, if you want to be more specific in the training process, try the following:
```
python3 src/main.py python src/main.py --mode=train --task=cartpole --population_size=200 --mutation_rate=0.07 --max_generations=20 --np=4 --crossover='uniform'
```

All these parameters are at your disposal. Note, that you can parallelize the training process by specifying the `--np` option.

To visualize one of the enclosed trained agent models, simply
use the following command:
```
make replay path=[path to the trained agent pickle]
make replay path=./agents/walker.pkl
make replay path=./agents/cartpole.pkl
```
