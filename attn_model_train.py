import numpy as np
import torch
from torch.utils.data import DataLoader

from lbipe.models import TorqueModel, AttnModel
from lbipe.losses import loss_fn_x
from lbipe.utils import pack_cut, generate_attn_dataset, combine_attn_dataset, x_by_pseudo

best_vloss_attn = 100000

tau1_scale = 0.44931670737585383
tau2_scale = 1.2213040989615254
tau3_scale = 0.6355422558900072
tau4_scale = 0.29493976426287805


def attn_train_loop(dataloader, torque_model, attn_model, optimizer):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    train_loss_m = 0
    train_loss_com = 0

    for batch, (sample, label, torques_g, A, x) in enumerate(dataloader):

        num_sample = len(sample)

        # reshape
        sample = torch.reshape(sample, (-1, 16))
        label = torch.reshape(label, (-1, 4))

        # torque model
        pred = torque_model(sample)

        # denorm
        tau1_est = pred[:, 0] * tau1_scale
        tau2_est = pred[:, 1] * tau2_scale
        tau3_est = pred[:, 2] * tau3_scale
        tau4_est = pred[:, 3] * tau4_scale
        torques_est = torch.stack((tau1_est, tau2_est, tau3_est, tau4_est), dim=1)

        # attn model
        w = attn_model(sample)

        # reshape
        torques_est = torch.reshape(torques_est, (num_sample, -1))
        w = torch.reshape(w, (num_sample, -1))

        # pseudo
        x_est = torch.zeros((num_sample, 4))
        for i in range(num_sample):
            x_est[i] = x_by_pseudo(torques_est[i] - torques_g[i], w[i], A[i])
        loss, loss_m, loss_com = loss_fn_x(x_est, x)

        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_m += loss_m.item()
        train_loss_com += loss_com.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * num_sample
            loss_m, loss_com = loss_m.item(), loss_com.item()
            print(f"loss: {loss:>7f}  loss_m: {loss_m:>7f}  loss_com: {loss_com:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    train_loss_m /= num_batches
    train_loss_com /= num_batches
    print(f"Avg loss: {train_loss:>7f}  loss_m: {train_loss_m:>7f}  loss_com: {train_loss_com:>7f}")


def attn_val_loop(dataloader, torque_model, attn_model, file_attn):

    global best_vloss_attn

    num_batches = len(dataloader)
    val_loss = 0
    val_loss_m = 0
    val_loss_com = 0

    with torch.no_grad():
        for sample, label, torques_g, A, x in dataloader:

            num_sample = len(sample)

            # reshape
            sample = torch.reshape(sample, (-1, 16))
            label = torch.reshape(label, (-1, 4))

            # torque model
            pred = torque_model(sample)

            # denorm
            tau1_est = pred[:, 0] * tau1_scale
            tau2_est = pred[:, 1] * tau2_scale
            tau3_est = pred[:, 2] * tau3_scale
            tau4_est = pred[:, 3] * tau4_scale
            torques_est = torch.stack((tau1_est, tau2_est, tau3_est, tau4_est), dim=1)

            # attn model
            w = attn_model(sample)

            # reshape
            torques_est = torch.reshape(torques_est, (num_sample, -1))
            w = torch.reshape(w, (num_sample, -1))

            # pseudo
            x_est = torch.zeros((num_sample, 4))
            for i in range(num_sample):
                x_est[i] = x_by_pseudo(torques_est[i] - torques_g[i], w[i], A[i])
            loss, loss_m, loss_com = loss_fn_x(x_est, x)

            val_loss += loss.item()
            val_loss_m += loss_m.item()
            val_loss_com += loss_com.item()

    val_loss /= num_batches
    val_loss_m /= num_batches
    val_loss_com /= num_batches
    print(f"Avg loss: {val_loss:>7f}  loss_m: {val_loss_m:>7f}  loss_com: {val_loss_com:>7f}")

    if val_loss < best_vloss_attn:
        best_vloss_attn = val_loss
        print('Save attn model!')
        torch.save(attn_model.state_dict(), file_attn)

    attn_model.load_state_dict(torch.load(file_attn))


def attn_model_train(class_torque, class_attn, file_torque, file_attn, sample_size, learning_rate, batch_size, epoches):

    # load and cut
    train_50g = np.load('data/data_train_50g.npz')
    train_100g = np.load('data/data_train_100g.npz')
    train_150g = np.load('data/data_train_150g.npz')

    train_random_50g = np.load('data/data_train_random_50g.npz')
    train_random_100g = np.load('data/data_train_random_100g.npz')
    train_random_150g = np.load('data/data_train_random_150g.npz')

    train_rd_50g = pack_cut(train_random_50g, np.arange(0, 9000))
    train_rd_100g = pack_cut(train_random_100g, np.arange(0, 9000))
    train_rd_150g = pack_cut(train_random_150g, np.arange(0, 9000))

    val_50g = pack_cut(train_random_50g, np.arange(9000, 10000))
    val_100g = pack_cut(train_random_100g, np.arange(9000, 10000))
    val_150g = pack_cut(train_random_150g, np.arange(9000, 10000))

    # dataset
    train_dataset_50g = generate_attn_dataset(train_50g, sample_size=sample_size, sample_num=11536)
    train_dataset_100g = generate_attn_dataset(train_100g, sample_size=sample_size, sample_num=11536)
    train_dataset_150g = generate_attn_dataset(train_150g, sample_size=sample_size, sample_num=11536)
    train_rd_dataset_50g = generate_attn_dataset(train_rd_50g, sample_size=sample_size, sample_num=9000)
    train_rd_dataset_100g = generate_attn_dataset(train_rd_100g, sample_size=sample_size, sample_num=9000)
    train_rd_dataset_150g = generate_attn_dataset(train_rd_150g, sample_size=sample_size, sample_num=9000)
    train_dataset = combine_attn_dataset([
        train_dataset_50g,
        train_dataset_100g,
        train_dataset_150g,
        train_rd_dataset_50g,
        train_rd_dataset_100g,
        train_rd_dataset_150g
    ])

    val_dataset_50g = generate_attn_dataset(val_50g, sample_size=sample_size, sample_num=1000)
    val_dataset_100g = generate_attn_dataset(val_100g, sample_size=sample_size, sample_num=1000)
    val_dataset_150g = generate_attn_dataset(val_150g, sample_size=sample_size, sample_num=1000)
    val_dataset = combine_attn_dataset([
        val_dataset_50g,
        val_dataset_100g,
        val_dataset_150g
    ])

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # train
    torque_model = class_torque()
    torque_model.load_state_dict(torch.load(file_torque))
    attn_model = class_attn()
    optimizer = torch.optim.Adam(attn_model.parameters(), lr=learning_rate)

    for t in range(epoches):
        print(f"\nEpoch {t + 1}\n-------------------------------")
        print('<train_loop>')
        attn_train_loop(train_dataloader, torque_model, attn_model, optimizer)
        print('<val_loop>')
        attn_val_loop(val_dataloader, torque_model, attn_model, file_attn)
    print('Done!')


def main():

    attn_model_train(
        class_torque=TorqueModel,
        class_attn=AttnModel,
        file_torque='dicts/dict_torque_new.pt',
        file_attn='dicts/dict_attn_new.pt',
        sample_size=64,
        learning_rate=1e-4,
        batch_size=32,
        epoches=30
    )


if __name__ == '__main__':

    main()
