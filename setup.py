from setuptools import setup, find_packages

setup(
    name='quarks2cosmos',
    version='0.0.1',
    author='EiffL',
    description='Utilities for data challenge',
    packages=find_packages(),    
    install_requires=['dm-haiku', 'lenspack', 'galsim', 'jax-cosmo', 'tensorflow-probability', 
                      'tensorflow-datasets', 'optax', 'flax']
)
