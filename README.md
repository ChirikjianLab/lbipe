# Learning-Based Inertial Properties Estimation

This is the code for the paper

> [**A Learning-Based Approach for Estimating Inertial Properties of Unknown Objects From Encoder Discrepancies**](https://ieeexplore.ieee.org/document/10176292)  
> Zizhou Lao, Yuanfeng Han, Yunshan Ma, Gregory S. Chirikjian

published on IEEE Robotics and Automation Letters (RA-L).

For a robot manipulator grasping an unknown object, the mass and center of mass can be identified without using force/torque sensors.

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
