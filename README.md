# Learning-Based Inertial Properties Estimation (lbipe)

This is the repo for paper:

> [**A Learning-Based Approach for Estimating Inertial Properties of Unknown Objects From Encoder Discrepancies**](https://ieeexplore.ieee.org/document/10176292)

published on IEEE Robotics and Automation Letters (RA-L).

For a robot manipulator grasping an unknown object, the mass and center of mass can be identified without using force/torque sensors.

## Introduction

A learning-based approach is proposed for estimating the mass and center of mass of unknown objects.
The framework requires only encoder measurements without using any force/torque sensors, but still maintains accurate estimation capability.
The block diagram is shown below:

<div align='center'>
  <img src='image/overview.pdf' width='1000px'>
</div>

(a) Training process. 
The training data are collected using known objects. 
The torque model is then trained using the derived ground truth of joint torque. 
Subsequently, the attention model is trained based on the trained torque model. 

(b) Testing process. 
The testing data are collected using unknown objects. 
The torque model is utilized to estimate the joint torque, while the attention model generates the weight matrix. 
The mass and COM of the unknown objects are solved by weighted least squares.

## Setup

All code was developed and tested on Ubuntu 20.04 with Python 3.9 and PyTorch 1.11. You are suggested to create a virtual environment in conda, activate the environment, and install the required .

```
conda create --name myenv  # create a vitual environment
conda activate myenv  # activate the virual environment
conda install --file requirements.txt  # install dependencies
conda deactivate  # exit the virtual environment
```

## Dataset

- `data/data_train_<mass>.npz`: 65000 samples with contains 10 pretrained models for all five datasets.
- `data/data_train_random_<mass>.npz`: 65000 samples with contains 10 pretrained models for all five datasets.
- `data/data_test_<object>.npz`: 1000 random samples with testing object.

## Running Pretrained Models

The results in the paper are generated with ten pairs of torque models and attention models are available in `dicts/`. 
You can use the scripts `torque_model_test.py` and `attn_model_test.py` to evaluate the torque model and attention model using pretrained models.

## Training New Models

You can use the scripts `torque_model_train.py` and `attn_model_train.py` to train new models using the provided dataset. 
It should be noted that a torque model should be trained first, as an attention model is trained based on a trained torque model.
