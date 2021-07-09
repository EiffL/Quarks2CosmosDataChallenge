# Quarks2Cosmos Deep Inverse Problems Data Challenge
Material for the CMU Quarks2Cosmos Conference Data Challenges

## Challenge Outline

This challenge is designed to be modular, so that you can follow one day and/or miss another day and still be able
to make the most of a given day.

### Day I: Differentiable Forward Models

Learning objectives:
- How to write a probabilistic forward model for galaxy images with Jax + TensorFlow Probability
- How to perform inference over a Jax model
- Bonus: Perform inference by Wiener filtering

Challenge Goals:
- Solve the inverse problem using a parametric galaxy model

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
