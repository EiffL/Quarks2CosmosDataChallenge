# Quarks2Cosmos Deep Inverse Problems Data Challenge
Material for the CMU Quarks2Cosmos Conference Data Challenges

## Challenge Outline

This Data Challenge is designed to help you learn a number of concepts and tools and is structured around 2 concrete problems:
- a "guided challenge" on solving inverse problems (inpainting, deconvolution, deblending) on HSC galaxy images
- an ["open challenge"](notebooks/MappingDarkMatterDataChallenge.ipynb) on solving a Dark Matter Mass-Mapping problem from real HSC weak gravitational lensing data.

Participants may choose to primarily follow the guided challenge where answers will be povided at the end of each day, 
and/or dive a little bit deeper into the open challenge where no answer exist :-) and where they can apply some of the 
methodology they have learned in the guided challenge.

The schedule of the challenge is designed to be modular, so that you can follow one day and/or miss another day and 
still be able to make the most of a given day, solutions will be provided for the guided challenge at the end of each day.

### Day I: Differentiable Forward Models

- [Notebook](notebooks/PartI-DifferentiableForwardModel.ipynb)

Learning objectives:
- How to write a probabilistic forward model for galaxy images with Jax + TensorFlow Probability
- How to optimize parameters of a Jax model

Challenge Goals:
- Write a forward model of ground-based galaxy images

### Day II: Learning a prior with Generative Models

Learning objectives:
- Write an Auto-Encoder in Jax+Haiku
- Build a Normalizing Flow in Jax+Haiku+TensorFlow Probability
- Bonus: Learn a prior by Denoising Score Matching

Challenge Goals:
- Build a generative model of galaxy morphology from Space-Based images

### Day III: Bringing it all together

Learning objectives:
- Solve inverse problem by MAP
- Learn how to sample from the posterior using Variational Inference
- Bonus: Learn to sample with SDE

Challenge Goals:
- Recover high-resolution posterior images for HSC galaxies
- Propose an inpainting model for masked regions in HSC galaxies
- Bonus: Demonstrate single band deblending!

### Day IV: Extra hacking time 

Day IV is dedicated to extra hacking time and producing some nice plots for the presentation on Day V.
Participants can use this time to work on the Mass-Mapping open challenge, continue to explore the galaxy image problem, and/or ask more questions!

## How to get setup at PSC on Bridges 2

We will primarily work on the OnDemand interface, to set up your python environment do the following from a command line on a GPU node. **Warning**: This assumes you have a fresh account at PSC, if not the case, ask challenge organizers to check it won't ruin your setup.
```bash
$ module load AI cuda
$ pip install --upgrade pip
$ pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
$ pip install --upgrade tensorflow
```
Then clone this repo and install the package and dependencies:
```bash
$ git clone https://github.com/EiffL/Quarks2CosmosDataChallenge.git
$ cd Quarks2CosmosDataChallenge
$ pip install --user -e .
```
Finally, add links to the data already available at PSC. From your `Quarks2CosmosDataChallenge` folder:
```bash
$ ln -s /ocean/projects/cis210053p/shared/deep_inverse data
$ ln -s /ocean/projects/cis210053p/shared/deep_inverse/tensorflow_datasets ~/
```
