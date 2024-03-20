import torch


def loss_fn_tau(pred, y):
    w = [1, 1, 1, 1]
    loss = 0
    for i in range(4):
        temp = pred[:, i] - y[:, i]
        temp = temp ** 2
        temp = torch.mean(temp)
        loss += temp * w[i]
    return loss


def loss_fn_x(x_est, x):

    w_x = [1.0, 0.3]

    m = x[:, 0]
    m_us = torch.unsqueeze(m, 1)
    com = x[:, 1:4] / m_us

    m_est = x_est[:, 0]
    com_est = x_est[:, 1:4] / m_us  # use m_ref here

    temp = (m_est - m) ** 2
    loss_m = torch.mean(temp)

    temp = (com_est - com) ** 2
    temp = torch.sum(temp, dim=1)
    loss_com = torch.mean(temp)

    loss_a = loss_m * w_x[0] + loss_com * w_x[1]

    return loss_a, loss_m, loss_com
