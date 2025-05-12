## Adaptive Corruption for Denoising Corrupted Data

This repository hosts the PyTorch implementation of an adaptive corruption regime for learning clean distributions from corrupted data. This implementation is inspired by the paper: [Ambient Diffusion: Learning Clean Distributions from Corrupted Data](https://arxiv.org/abs/2305.19256).

Authored by: Abinitha Gourabathina, Supriya Lall

## Installation

The recommended way to run the code is with a Python virtual environment.

First, clone the repository: 

`git clone https://github.com/abinithago/68300-final.git`.

Then, create a new Python3 virtual environment:

`python -m venv venv`
`source venv/bin/activate`

Once you activate the virtual environment, run the below.

`pip3 install requirements.txt`


## Running Experiments

To run the CIFAR-10 experiments, activate the virtual environment and run the `cifar-10.ipynb` notebook.

Similarly, the relevant experiments for the MNIST datasets should be available in the `mnist_final.ipynb` notebook.
