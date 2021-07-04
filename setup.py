from setuptools import setup, find_packages

setup(
    name='quarks2cosmos',
    version='0.0.1',
    author='EiffL',
    description='Utilities for data challenge',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1'],
)