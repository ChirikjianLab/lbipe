import numpy as np
import torch
from torch.utils.data import Dataset

q1_min = np.deg2rad(-30)
q1_max = np.deg2rad(30)
q2_min = np.deg2rad(-50)
q2_max = np.deg2rad(-10)
q3_min = np.deg2rad(10)
q3_max = np.deg2rad(50)
q4_min = np.deg2rad(-20)
q4_max = np.deg2rad(20)

qe1_scale = 0.03761652711106622
qe2_scale = 0.07762744017531897
qe3_scale = 0.056842050482615925
qe4_scale = 0.039233182199103656

tau1_scale = 0.44931670737585383
tau2_scale = 1.2213040989615254
tau3_scale = 0.6355422558900072
tau4_scale = 0.29493976426287805


class TorqueDataset(Dataset):

    def __init__(self, sample, label):
        self.sample = sample
        self.label = label

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx], self.label[idx]


class AttnDataset(Dataset):

    def __init__(self, sample, label, torques_g, A, x):
        self.sample = sample
        self.label = label
        self.torques_g = torques_g
        self.A = A
        self.x = x

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx], self.label[idx], self.torques_g[idx], self.A[idx], self.x[idx]


def pack_cut(pack, idx):

    joint_data = pack['joint_data']
    tau = pack['tau']
    A = pack['A']
    x = pack['x']

    tau = np.reshape(tau, (-1, 4))
    A = np.reshape(A, (-1, 16))

    joint_data_cut = joint_data[idx]
    tau_cut = tau[idx]
    A_cut = A[idx]
    x_cut = x

    tau_cut = np.reshape(tau_cut, (-1,))
    A_cut = np.reshape(A_cut, (-1,))

    pack_cut = {
        'joint_data': joint_data_cut,
        'tau': tau_cut,
        'A': A_cut,
        'x': x_cut
    }

    return pack_cut


def generate_torque_dataset(joint_data):

    num_data = np.shape(joint_data)[0]

    # extract
    positions_actual = joint_data[:, 1, :]
    positions_error = joint_data[:, 2, :]
    directions = joint_data[:, 5, :]
    torques_derive = joint_data[:, 6, :]

    # directions
    # from [1, -1] to [(1, 0), (0, 1)]
    dir1 = np.zeros((num_data, 2))
    dir2 = np.zeros((num_data, 2))
    dir3 = np.zeros((num_data, 2))
    dir4 = np.zeros((num_data, 2))

    for idx_data in range(num_data):
        # joint 1
        if directions[idx_data, 0] > 0:
            dir1[idx_data, 0] = 1
        elif directions[idx_data, 0] < 0:
            dir1[idx_data, 1] = 1
        # joint 2
        if directions[idx_data, 1] > 0:
            dir2[idx_data, 0] = 1
        elif directions[idx_data, 1] < 0:
            dir2[idx_data, 1] = 1
        # joint 3
        if directions[idx_data, 2] > 0:
            dir3[idx_data, 0] = 1
        elif directions[idx_data, 2] < 0:
            dir3[idx_data, 1] = 1
        # joint 4
        if directions[idx_data, 3] > 0:
            dir4[idx_data, 0] = 1
        elif directions[idx_data, 3] < 0:
            dir4[idx_data, 1] = 1

    # qe, q, tau normalization
    # joint 1
    q1 = (positions_actual[:, 0] - q1_min) / (q1_max - q1_min)
    qe1 = positions_error[:, 0] / qe1_scale
    tau1 = torques_derive[:, 0] / tau1_scale
    # joint 2
    q2 = (positions_actual[:, 1] - q2_min) / (q2_max - q2_min)
    qe2 = positions_error[:, 1] / qe2_scale
    tau2 = torques_derive[:, 1] / tau2_scale
    # joint 3
    q3 = (positions_actual[:, 2] - q3_min) / (q3_max - q3_min)
    qe3 = positions_error[:, 2] / qe3_scale
    tau3 = torques_derive[:, 2] / tau3_scale
    # joint 4
    q4 = (positions_actual[:, 3] - q4_min) / (q4_max - q4_min)
    qe4 = positions_error[:, 3] / qe4_scale
    tau4 = torques_derive[:, 3] / tau4_scale

    # stack
    s1 = np.stack([q1, qe1], axis=1)
    s2 = np.stack([q2, qe2], axis=1)
    s3 = np.stack([q3, qe3], axis=1)
    s4 = np.stack([q4, qe4], axis=1)
    s1 = np.concatenate([s1, dir1], axis=1)
    s2 = np.concatenate([s2, dir2], axis=1)
    s3 = np.concatenate([s3, dir3], axis=1)
    s4 = np.concatenate([s4, dir4], axis=1)
    sample = np.concatenate([s1, s2, s3, s4], axis=1)
    label = np.stack([tau1, tau2, tau3, tau4], axis=1)

    # dataset
    sample = torch.from_numpy(sample).float()
    label = torch.from_numpy(label).float()
    torque_dataset = TorqueDataset(sample, label)

    return torque_dataset


def generate_attn_dataset(pack, sample_size, sample_num):

    # unpack
    joint_data = pack['joint_data']
    A = pack['A']
    x = pack['x']

    # num
    num_data = np.shape(joint_data)[0]

    # joint state
    # extract
    positions_actual = joint_data[:, 1, :]
    positions_error = joint_data[:, 2, :]
    directions = joint_data[:, 5, :]
    torques_derive = joint_data[:, 6, :]
    torques_g = joint_data[:, 7, :]

    # directions
    dir1 = np.zeros((num_data, 2))
    dir2 = np.zeros((num_data, 2))
    dir3 = np.zeros((num_data, 2))
    dir4 = np.zeros((num_data, 2))
    for idx_data in range(num_data):
        # joint 1
        if directions[idx_data, 0] > 0:
            dir1[idx_data, 0] = 1
        elif directions[idx_data, 0] < 0:
            dir1[idx_data, 1] = 1
        # joint 2
        if directions[idx_data, 1] > 0:
            dir2[idx_data, 0] = 1
        elif directions[idx_data, 1] < 0:
            dir2[idx_data, 1] = 1
        # joint 3
        if directions[idx_data, 2] > 0:
            dir3[idx_data, 0] = 1
        elif directions[idx_data, 2] < 0:
            dir3[idx_data, 1] = 1
        # joint 4
        if directions[idx_data, 3] > 0:
            dir4[idx_data, 0] = 1
        elif directions[idx_data, 3] < 0:
            dir4[idx_data, 1] = 1

    # qe, q, tau normalization
    # joint 1
    q1 = (positions_actual[:, 0] - q1_min) / (q1_max - q1_min)
    qe1 = positions_error[:, 0] / qe1_scale
    tau1 = torques_derive[:, 0] / tau1_scale
    # joint 2
    q2 = (positions_actual[:, 1] - q2_min) / (q2_max - q2_min)
    qe2 = positions_error[:, 1] / qe2_scale
    tau2 = torques_derive[:, 1] / tau2_scale
    # joint 3
    q3 = (positions_actual[:, 2] - q3_min) / (q3_max - q3_min)
    qe3 = positions_error[:, 2] / qe3_scale
    tau3 = torques_derive[:, 2] / tau3_scale
    # joint 4
    q4 = (positions_actual[:, 3] - q4_min) / (q4_max - q4_min)
    qe4 = positions_error[:, 3] / qe4_scale
    tau4 = torques_derive[:, 3] / tau4_scale

    # stack
    s1 = np.stack([q1, qe1], axis=1)
    s2 = np.stack([q2, qe2], axis=1)
    s3 = np.stack([q3, qe3], axis=1)
    s4 = np.stack([q4, qe4], axis=1)
    s1 = np.concatenate([s1, dir1], axis=1)
    s2 = np.concatenate([s2, dir2], axis=1)
    s3 = np.concatenate([s3, dir3], axis=1)
    s4 = np.concatenate([s4, dir4], axis=1)
    sample = np.concatenate([s1, s2, s3, s4], axis=1)
    label = np.stack([tau1, tau2, tau3, tau4], axis=1)

    # to torch
    sample = torch.from_numpy(sample).float()
    label = torch.from_numpy(label).float()
    torques_g = torch.from_numpy(torques_g).float()
    A = torch.from_numpy(A).float()
    A = torch.reshape(A, (-1, 16))
    x = torch.from_numpy(x).float()

    # generate sample_num samples, which includes sample_size data
    dataset_sample = torch.zeros((sample_num, sample_size * 16))
    dataset_label = torch.zeros((sample_num, sample_size * 4))
    dataset_torques_g = torch.zeros((sample_num, sample_size * 4))
    dataset_A = torch.zeros((sample_num, sample_size * 16))

    rng = np.random.default_rng()
    for i in range(sample_num):
        idx = rng.choice(num_data, size=sample_size, replace=False)
        dataset_sample[i] = torch.reshape(sample[idx], (-1,))
        dataset_label[i] = torch.reshape(label[idx], (-1,))
        dataset_torques_g[i] = torch.reshape(torques_g[idx], (-1,))
        dataset_A[i] = torch.reshape(A[idx], (-1,))
    dataset_x = x.repeat(sample_num, 1)

    # dataset
    attn_dataset = AttnDataset(dataset_sample, dataset_label, dataset_torques_g, dataset_A, dataset_x)

    return attn_dataset


def combine_attn_dataset(dataset_list):

    sample_list = []
    label_list = []
    torques_g_list = []
    A_list = []
    x_list = []

    for i in range(len(dataset_list)):
        sample_list.append(dataset_list[i].sample)
        label_list.append(dataset_list[i].label)
        torques_g_list.append(dataset_list[i].torques_g)
        A_list.append(dataset_list[i].A)
        x_list.append(dataset_list[i].x)

    sample = torch.cat(sample_list, dim=0)
    label = torch.cat(label_list, dim=0)
    torques_g = torch.cat(torques_g_list, dim=0)
    A = torch.cat(A_list, dim=0)
    x = torch.cat(x_list, dim=0)

    attn_dataset = AttnDataset(sample, label, torques_g, A, x)

    return attn_dataset


def torque_by_model(file_torque, class_torque, joint_data):

    torque_dataset = generate_torque_dataset(joint_data)
    torque_model = class_torque()
    torque_model.load_state_dict(torch.load(file_torque))
    torque_model.eval()

    pred = torque_model(torque_dataset.sample)
    tau1_est = (pred[:, 0] * tau1_scale).detach().numpy()
    tau2_est = (pred[:, 1] * tau2_scale).detach().numpy()
    tau3_est = (pred[:, 2] * tau3_scale).detach().numpy()
    tau4_est = (pred[:, 3] * tau4_scale).detach().numpy()
    tau_est = np.stack((tau1_est, tau2_est, tau3_est, tau4_est), axis=1)

    return tau_est


def x_by_pseudo(tau, w, A):

    W = torch.diag(w, 0)
    A = torch.reshape(A, (-1, 4))

    JM = torch.t(A) @ W @ A
    JM = torch.linalg.inv(JM)
    JM = JM @ torch.t(A) @ W
    x = JM @ tau

    return x
