import numpy as np
from scipy.optimize import curve_fit

from lbipe.utils import pack_cut


def func_sensor(x, tau_f, const):
    return x[0] - x[1] * tau_f + const


def func_qe(x, k, tau_f, const):
    return k * x[0] - x[1] * tau_f + const


def fit_params_sensor():

    train3_0g = np.load('data/data_train_0g.npz')
    train3_50g = np.load('data/data_train_50g.npz')
    train3_100g = np.load('data/data_train_100g.npz')
    train3_150g = np.load('data/data_train_150g.npz')

    train3random_0g = np.load('data/data_train_random_0g.npz')
    train3random_50g = np.load('data/data_train_random_50g.npz')
    train3random_100g = np.load('data/data_train_random_100g.npz')
    train3random_150g = np.load('data/data_train_random_150g.npz')

    train3rd_0g = pack_cut(train3random_0g, np.arange(0, 9000))
    train3rd_50g = pack_cut(train3random_50g, np.arange(0, 9000))
    train3rd_100g = pack_cut(train3random_100g, np.arange(0, 9000))
    train3rd_150g = pack_cut(train3random_150g, np.arange(0, 9000))

    train3_data_0g = train3_0g['joint_data']
    train3_data_50g = train3_50g['joint_data']
    train3_data_100g = train3_100g['joint_data']
    train3_data_150g = train3_150g['joint_data']

    train3rd_data_0g = train3rd_0g['joint_data']
    train3rd_data_50g = train3rd_50g['joint_data']
    train3rd_data_100g = train3rd_100g['joint_data']
    train3rd_data_150g = train3rd_150g['joint_data']

    train3_data = np.concatenate(
        [
            train3_data_0g,
            train3_data_50g,
            train3_data_100g,
            train3_data_150g,
            train3rd_data_0g,
            train3rd_data_50g,
            train3rd_data_100g,
            train3rd_data_150g
        ],
        axis=0
    )

    params = np.zeros((4, 2))
    for i in range(4):
        x = train3_data[:, [4, 5], i]
        x = np.transpose(x)
        y = train3_data[:, 6, i]
        popt, _ = curve_fit(func_sensor, x, y)
        # print(i, popt)
        params[i] = popt

    return params


def fit_params_qe():

    train3_0g = np.load('data/data_train_0g.npz')
    train3_50g = np.load('data/data_train_50g.npz')
    train3_100g = np.load('data/data_train_100g.npz')
    train3_150g = np.load('data/data_train_150g.npz')

    train3random_0g = np.load('data/data_train_random_0g.npz')
    train3random_50g = np.load('data/data_train_random_50g.npz')
    train3random_100g = np.load('data/data_train_random_100g.npz')
    train3random_150g = np.load('data/data_train_random_150g.npz')

    train3rd_0g = pack_cut(train3random_0g, np.arange(0, 9000))
    train3rd_50g = pack_cut(train3random_50g, np.arange(0, 9000))
    train3rd_100g = pack_cut(train3random_100g, np.arange(0, 9000))
    train3rd_150g = pack_cut(train3random_150g, np.arange(0, 9000))

    train3_data_0g = train3_0g['joint_data']
    train3_data_50g = train3_50g['joint_data']
    train3_data_100g = train3_100g['joint_data']
    train3_data_150g = train3_150g['joint_data']

    train3rd_data_0g = train3rd_0g['joint_data']
    train3rd_data_50g = train3rd_50g['joint_data']
    train3rd_data_100g = train3rd_100g['joint_data']
    train3rd_data_150g = train3rd_150g['joint_data']

    train3_data = np.concatenate(
        [
            train3_data_0g,
            train3_data_50g,
            train3_data_100g,
            train3_data_150g,
            train3rd_data_0g,
            train3rd_data_50g,
            train3rd_data_100g,
            train3rd_data_150g
        ],
        axis=0
    )

    params = np.zeros((4, 3))
    for i in range(4):
        x = train3_data[:, [2, 5], i]
        x = np.transpose(x)
        y = train3_data[:, 6, i]
        popt, _ = curve_fit(func_qe, x, y)
        # print(i, popt)
        params[i] = popt

    return params


def torque_by_sensor(joint_data):

    params = fit_params_sensor()
    tau = np.zeros((len(joint_data), 4))
    for i in range(4):
        x = joint_data[:, [4, 5], i]
        x = np.transpose(x)
        popt = params[i]
        y = func_sensor(x, popt[0], popt[1])
        tau[:, i] = y

    return tau


def torque_by_qe(joint_data):

    params = fit_params_qe()
    tau = np.zeros((len(joint_data), 4))
    for i in range(4):
        x = joint_data[:, [2, 5], i]
        x = np.transpose(x)
        popt = params[i]
        y = func_qe(x, popt[0], popt[1], popt[2])
        tau[:, i] = y

    return tau
