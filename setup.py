#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='dcgan',
    version='0.1.0',
    description='Implementation of Deep Convolutional GAN using PyTorch',
    author='Abhiroop Tejomay',
    author_email='abhirooptejomay@gmail.com',
    url='https://github.com/visualCalculus/deep-convolutional-gan',
    packages=find_packages(include=["dcgan", "dcgan.*"]),
)