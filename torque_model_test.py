import numpy as np

from lbipe.models import TorqueModel
from lbipe.utils import torque_by_model
from lbipe.baselines import torque_by_sensor, torque_by_qe


def torque_model_test(class_torque, file_torque_list):

    # test data
    test_cube = np.load('data/data_test_cube.npz')
    test_red = np.load('data/data_test_red.npz')
    test_white = np.load('data/data_test_white.npz')
    test_black = np.load('data/data_test_black.npz')

    test_data_cube = test_cube['joint_data']
    test_data_red = test_red['joint_data']
    test_data_white = test_white['joint_data']
    test_data_black = test_black['joint_data']

    test_data = np.concatenate(
        [
            test_data_cube,
            test_data_red,
            test_data_white,
            test_data_black
        ],
        axis=0
    )

    # results of multiple torque models
    num_data = len(test_data)
    num_model = len(file_torque_list)
    tau_model = np.zeros((num_data * num_model, 4))
    for i in range(num_model):
        tau_model[num_data * i : num_data * (i + 1)] = torque_by_model(file_torque_list[i], class_torque, test_data)

    # error of tau_model
    tau_gt = np.tile(test_data[:, 6, :], (num_model, 1))
    tau_max = np.amax(np.abs(tau_gt), axis=0)
    AE_model = np.abs(tau_model - tau_gt)
    MAE_model = np.mean(AE_model, axis=0)
    NMAE_model = MAE_model / tau_max
    NRMSE_model = np.sqrt(np.sum(AE_model ** 2, axis=0) / (num_data * num_model)) / tau_max

    # results of sensor and pe
    tau_sensor = torque_by_sensor(test_data)
    tau_pe = torque_by_qe(test_data)

    # error of sensor and pe
    tau_gt = test_data[:, 6, :]
    tau_max = np.amax(np.abs(tau_gt), axis=0)
    AE_sensor = np.abs(tau_sensor - tau_gt)
    MAE_sensor = np.mean(AE_sensor, axis=0)
    NMAE_sensor = MAE_sensor / tau_max
    NRMSE_sensor = np.sqrt(np.sum(AE_sensor ** 2, axis=0) / num_data) / tau_max
    AE_pe = np.abs(tau_pe - tau_gt)
    MAE_pe = np.mean(AE_pe, axis=0)
    NMAE_pe = MAE_pe / tau_max
    NRMSE_pe = np.sqrt(np.sum(AE_pe ** 2, axis=0) / num_data) / tau_max

    print(f'-------- model --------')
    for idx_joint in range(4):
        print(f'[joint{idx_joint + 1}] MAE: {MAE_model[idx_joint] * 1000:.2f}Nmm, NMAE: {NMAE_model[idx_joint] * 100:.2f}%, NRMSE: {NRMSE_model[idx_joint] * 100:.2f}%')
    print(f'[average] MAE: {np.mean(MAE_model) * 1000:.2f}Nmm, NMAE: {np.mean(NMAE_model) * 100:.2f}%, NRMSE: {np.mean(NRMSE_model) * 100:.2f}%')

    print(f'-------- sensor --------')
    for idx_joint in range(4):
        print(f'[joint{idx_joint + 1}] MAE: {MAE_sensor[idx_joint] * 1000:.2f}Nmm, NMAE: {NMAE_sensor[idx_joint] * 100:.2f}%, NRMSE: {NRMSE_sensor[idx_joint] * 100:.2f}%')
    print(f'[average] MAE: {np.mean(MAE_sensor) * 1000:.2f}Nmm, NMAE: {np.mean(NMAE_sensor) * 100:.2f}%, NRMSE: {np.mean(NRMSE_sensor) * 100:.2f}%')

    print(f'-------- pe --------')
    for idx_joint in range(4):
        print(f'[joint{idx_joint + 1}] MAE: {MAE_pe[idx_joint] * 1000:.2f}Nmm, NMAE: {NMAE_pe[idx_joint] * 100:.2f}%, NRMSE: {NRMSE_pe[idx_joint] * 100:.2f}%')
    print(f'[average] MAE: {np.mean(MAE_pe) * 1000:.2f}Nmm, NMAE: {np.mean(NMAE_pe) * 100:.2f}%, NRMSE: {np.mean(NRMSE_pe) * 100:.2f}%')


def main():

    dicts_torque = [
        'dicts/dict_torque_1.pt',
        'dicts/dict_torque_2.pt',
        'dicts/dict_torque_3.pt',
        'dicts/dict_torque_4.pt',
        'dicts/dict_torque_5.pt',
        'dicts/dict_torque_6.pt',
        'dicts/dict_torque_7.pt',
        'dicts/dict_torque_8.pt',
        'dicts/dict_torque_9.pt',
        'dicts/dict_torque_10.pt'
    ]

    torque_model_test(
        class_torque=TorqueModel,
        file_torque_list=dicts_torque
    )


if __name__ == '__main__':

    main()
