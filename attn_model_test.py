import numpy as np
import torch

from lbipe.models import TorqueModel, AttnModel
from lbipe.utils import torque_by_model, generate_torque_dataset, x_by_pseudo
from lbipe.baselines import torque_by_sensor, torque_by_qe


def estimate_object(class_torque, class_attn, file_torque_list, file_attn_list, pack, L_max):

    # unpack
    joint_data = pack['joint_data']
    A = pack['A']
    x_gt = pack['x']

    # torque
    tau_g = joint_data[:, 7, :]
    tau_gt = joint_data[:, 6, :]
    tau_sensor = torque_by_sensor(joint_data)
    tau_qe = torque_by_qe(joint_data)

    # to torch and reshape for random
    tau_g = torch.from_numpy(tau_g).float()
    tau_gt = torch.from_numpy(tau_gt).float()
    tau_sensor = torch.from_numpy(tau_sensor).float()
    tau_qe = torch.from_numpy(tau_qe).float()
    A = torch.from_numpy(A).float()
    A = torch.reshape(A, (-1, 16))

    # apply torque model and attn model
    num_model = len(file_torque_list)
    tau_model_list = []
    w_list = []
    torque_dataset = generate_torque_dataset(joint_data)
    sample = torque_dataset.sample
    for idx_model in range(num_model):
        # torque model
        tau_model = torque_by_model(file_torque_list[idx_model], class_torque, joint_data)
        tau_model = torch.from_numpy(tau_model).float()
        tau_model_list.append(tau_model)
        # attn model
        attn_model = class_attn()
        attn_model.load_state_dict(torch.load(file_attn_list[idx_model]))
        attn_model.eval()
        w_list.append(attn_model(sample))

    # random test
    num_data = len(joint_data)
    sample_size = 64
    sample_num = 1000
    x_sensor = torch.zeros((sample_num, 4))
    x_qe = torch.zeros((sample_num, 4))
    x_t = torch.zeros((num_model, sample_num, 4))
    x_ta = torch.zeros((num_model, sample_num, 4))

    rng = np.random.default_rng()
    for i in range(sample_num):
        # random sample
        idx = rng.choice(num_data, size=sample_size, replace=False)

        # baseline
        tau_g_s = tau_g[idx]  # this sample is different from the previous sample
        tau_sensor_s = tau_sensor[idx]
        tau_qe_s = tau_qe[idx]
        A_s = A[idx]
        w_iden_s = torch.ones((sample_size, 4))

        tau_g_s = torch.reshape(tau_g_s, (-1,))
        tau_sensor_s = torch.reshape(tau_sensor_s, (-1,))
        tau_qe_s = torch.reshape(tau_qe_s, (-1,))
        A_s = torch.reshape(A_s, (-1,))
        w_iden_s = torch.reshape(w_iden_s, (-1,))

        x_sensor[i] = x_by_pseudo(tau_sensor_s - tau_g_s, w_iden_s, A_s)
        x_qe[i] = x_by_pseudo(tau_qe_s - tau_g_s, w_iden_s, A_s)

        # model
        for j in range(num_model):
            tau_model = tau_model_list[j]
            tau_model_s = tau_model[idx]
            tau_model_s = torch.reshape(tau_model_s, (-1,))

            w = w_list[j]
            w_s = w[idx]
            w_s = torch.reshape(w_s, (-1,))

            x_t[j, i] = x_by_pseudo(tau_model_s - tau_g_s, w_iden_s, A_s)
            x_ta[j, i] = x_by_pseudo(tau_model_s - tau_g_s, w_s, A_s)

    # to numpy
    x_sensor = x_sensor.detach().numpy()
    x_qe = x_qe.detach().numpy()
    x_t = x_t.detach().numpy()
    x_ta = x_ta.detach().numpy()

    x_t = np.reshape(x_t, (-1, 4))
    x_ta = np.reshape(x_ta, (-1, 4))

    # print w mean
    w = torch.cat(w_list, dim=0)
    w = w.detach().numpy()
    w_mean = np.mean(w, axis=0)
    print(f"w: {w_mean}")

    # error
    m_gt = x_gt[0]
    com_gt = x_gt[1:4] / m_gt

    # print and plot
    x_list = [x_sensor, x_qe, x_t, x_ta]
    name_list = ['sensor', 'qe', 't', 't-a']
    num_method = len(x_list)

    m_metric = np.zeros((4, 3))
    com_metric = np.zeros((4, 3))

    for i in range(num_method):
        x_est = x_list[i]

        # est
        m_est = x_est[:, 0]
        com_est = x_est[:, 1:4] / np.expand_dims(m_est, axis=1)

        # error
        n = len(x_est)
        AE_m = np.abs(m_est - m_gt)
        MAE_m = np.mean(AE_m)
        NMAE_m = MAE_m / m_gt
        NRMSE_m = np.sqrt(np.sum(AE_m ** 2) / n) / m_gt
        AE_com = np.sqrt(np.sum((com_est - com_gt) ** 2, axis=1))
        MAE_com = np.mean(AE_com)
        NMAE_com = MAE_com / L_max
        NRMSE_com = np.sqrt(np.sum(AE_com ** 2) / n) / L_max

        m_metric[i, 0] = MAE_m
        m_metric[i, 1] = NMAE_m
        m_metric[i, 2] = NRMSE_m
        com_metric[i, 0] = MAE_com
        com_metric[i, 1] = NMAE_com
        com_metric[i, 2] = NRMSE_com

        # print
        print(f'-------- {name_list[i]} --------')
        print(f'[mass error] MAE: {MAE_m * 1000:.2f}g, NMAE: {NMAE_m * 100:.2f}%, NRMSE: {NRMSE_m * 100:.2f}%.')
        print(f'[com error] MAE: {MAE_com * 1000:.1f}mm, NMAE: {NMAE_com * 100:.2f}%, NRMSE: {NRMSE_com * 100:.2f}%.')

    return m_metric, com_metric


def attn_model_test(class_torque, class_attn, file_torque_list, file_attn_list):

    test_cube = np.load('data/data_test_cube.npz')
    test_red = np.load('data/data_test_red.npz')
    test_white = np.load('data/data_test_white.npz')
    test_black = np.load('data/data_test_black.npz')

    mass_array = np.zeros((4, 4, 3))  # (object, method, metric)
    com_array = np.zeros((4, 4, 3))

    print('======== cube ========')
    mass_array[0], com_array[0] = estimate_object(
        class_torque=class_torque,
        class_attn=class_attn,
        file_torque_list=file_torque_list,
        file_attn_list=file_attn_list,
        pack=test_cube,
        L_max=69.2820 / 1000
    )
    print('======== red ========')
    mass_array[1], com_array[1] = estimate_object(
        class_torque=class_torque,
        class_attn=class_attn,
        file_torque_list=file_torque_list,
        file_attn_list=file_attn_list,
        pack=test_red,
        L_max=66.5808 / 1000
    )
    print('======== white ========')
    mass_array[2], com_array[2] = estimate_object(
        class_torque=class_torque,
        class_attn=class_attn,
        file_torque_list=file_torque_list,
        file_attn_list=file_attn_list,
        pack=test_white,
        L_max=75.5513 / 1000
    )
    print('======== black ========')
    mass_array[3], com_array[3] = estimate_object(
        class_torque=class_torque,
        class_attn=class_attn,
        file_torque_list=file_torque_list,
        file_attn_list=file_attn_list,
        pack=test_black,
        L_max=85.1880 / 1000
    )

    mass_metric_mean = np.mean(mass_array, axis=0)
    com_metric_mean = np.mean(com_array, axis=0)
    print('======== average ========')
    name_list = ['sensor', 'qe', 't', 't-a']
    for i in range(4):
        print(f'-------- {name_list[i]} --------')
        print(
            f'[mass error] MAE: {mass_metric_mean[i, 0] * 1000:.2f}g, NMAE: {mass_metric_mean[i, 1] * 100:.2f}%, NRMSE: {mass_metric_mean[i, 2] * 100:.2f}%.')
        print(
            f'[com error] MAE: {com_metric_mean[i, 0] * 1000:.1f}mm, NMAE: {com_metric_mean[i, 1] * 100:.2f}%, NRMSE: {com_metric_mean[i, 2] * 100:.2f}%.')


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

    dicts_attn = [
        'dicts/dict_attn_1.pt',
        'dicts/dict_attn_2.pt',
        'dicts/dict_attn_3.pt',
        'dicts/dict_attn_4.pt',
        'dicts/dict_attn_5.pt',
        'dicts/dict_attn_6.pt',
        'dicts/dict_attn_7.pt',
        'dicts/dict_attn_8.pt',
        'dicts/dict_attn_9.pt',
        'dicts/dict_attn_10.pt'
    ]

    attn_model_test(
        class_torque=TorqueModel,
        class_attn=AttnModel,
        file_torque_list=dicts_torque,
        file_attn_list=dicts_attn
    )


if __name__ == '__main__':

    main()
