# ANN-hyper-parameter-optimisation-with-GA
This library descires an algorithm which creates neural networks and improves them until the designated goal is reached. The algorithm creates a population of diferent networks and continiously produces generations of networks and applies natural selection to extinct the bad individuals while protecting the best ones. This code is for https://www.kaggle.com/c/pubg-finish-placement-prediction competition. 


A genetic algorithm (GA) designed to optimise the hyper parameters of an neural network as well as evolving its structure.
The algorithm supports steady state GA and Generational GA.
Cassification and regression, minimisation and maximisation options are added.
Algorithm favours inovations and protect inovators for 5 generations. This feature increases the diversity of the population.
Roulette wheel selection is used. 
