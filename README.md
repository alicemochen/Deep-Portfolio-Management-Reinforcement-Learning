# Study - A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem

This repo is my study of Deep Reinforcement Learning. Many of the code pieces are from the following original repo.
I make changes to the code to make it a bit more adaptive. It takes in a list of names,
start date, and end date as params, download data, and then run the agent. 

## To run:

Run main.py. The program takes in directories as input, read in a config file from each directory,
and store output files into the same directory. I created 2 available configs in
run_directory.

## Next steps

- I will rewrite the model using keras. I will recreate this myself, moving away
from the original code base
- I would also try to add in Bellman equation instead of just policy gradient.
- I will maintain 2 networks, one for critic, one for action
- I will try to evaluate the policy (mean/variance)


## Original paper
This original project is an implementation and further research of the original paper [A Deep Reinforcement Learning Framework for the
Financial Portfolio Management Problem (Jiang et al. 2017)](https://arxiv.org/abs/1706.10059). 

## Original Repo
https://github.com/selimamrouni/Deep-Portfolio-Management-Reinforcement-Learning

## Original Author

* **Selim Amrouni** [selimamrouni](https://github.com/selimamrouni)
* **Aymeric Moulin** [AymericCAMoulin](https://github.com/AymericCAMoulin)
* **Philippe Mizrahi** [pcamizrahi](https://github.com/pcamizrahi)





