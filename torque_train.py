import numpy as np
import torch
from torch.utils.data import DataLoader

from lbipe.models import TorqueModel
from lbipe.losses import loss_fn_tau
from lbipe.utils import pack_cut, generate_torque_dataset

best_vloss_torque = 100000


def torque_train_loop(dataloader, model, optimizer):
    print('<train_loop>')
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn_tau(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    print(f"Avg loss: {train_loss:>8f}")


def torque_val_loop(dataloader, model, file_torque):
    global best_vloss_torque

    print('<val_loop>')
    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            val_loss += loss_fn_tau(pred, y).item()

    val_loss /= num_batches
    print(f"Avg loss: {val_loss:>8f}")

    if val_loss < best_vloss_torque:
        best_vloss_torque = val_loss
        print('Save torque model!')
        torch.save(model.state_dict(), file_torque)

    model.load_state_dict(torch.load(file_torque))


def torque_model_train(class_torque, file_torque, learning_rate, batch_size, epoches):

    # load and cut
    train_0g = np.load('data/data_train_0g.npz')
    train_50g = np.load('data/data_train_50g.npz')
    train_100g = np.load('data/data_train_100g.npz')
    train_150g = np.load('data/data_train_150g.npz')

    train_random_0g = np.load('data/data_train_random_0g.npz')
    train_random_50g = np.load('data/data_train_random_50g.npz')
    train_random_100g = np.load('data/data_train_random_100g.npz')
    train_random_150g = np.load('data/data_train_random_150g.npz')

    train_rd_0g = pack_cut(train_random_0g, np.arange(0, 9000))
    train_rd_50g = pack_cut(train_random_50g, np.arange(0, 9000))
    train_rd_100g = pack_cut(train_random_100g, np.arange(0, 9000))
    train_rd_150g = pack_cut(train_random_150g, np.arange(0, 9000))

    val_0g = pack_cut(train_random_0g, np.arange(9000, 10000))
    val_50g = pack_cut(train_random_50g, np.arange(9000, 10000))
    val_100g = pack_cut(train_random_100g, np.arange(9000, 10000))
    val_150g = pack_cut(train_random_150g, np.arange(9000, 10000))

    # data
    train_data_0g = train_0g['joint_data']
    train_data_50g = train_50g['joint_data']
    train_data_100g = train_100g['joint_data']
    train_data_150g = train_150g['joint_data']

    train_rd_data_0g = train_rd_0g['joint_data']
    train_rd_data_50g = train_rd_50g['joint_data']
    train_rd_data_100g = train_rd_100g['joint_data']
    train_rd_data_150g = train_rd_150g['joint_data']

    train_data = np.concatenate(
        [
            train_data_0g,
            train_data_50g,
            train_data_100g,
            train_data_150g,
            train_rd_data_0g,
            train_rd_data_50g,
            train_rd_data_100g,
            train_rd_data_150g
        ],
        axis=0
    )

    val_data_0g = val_0g['joint_data']
    val_data_50g = val_50g['joint_data']
    val_data_100g = val_100g['joint_data']
    val_data_150g = val_150g['joint_data']

    val_data = np.concatenate(
        [
            val_data_0g,
            val_data_50g,
            val_data_100g,
            val_data_150g
        ],
        axis=0
    )

    # dataset
    train_dataset = generate_torque_dataset(train_data)
    val_dataset = generate_torque_dataset(val_data)

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # train
    torque_model = class_torque()
    optimizer = torch.optim.Adam(torque_model.parameters(), lr=learning_rate)

    for t in range(epoches):
        print(f"\nEpoch {t + 1}\n-------------------------------")
        torque_train_loop(train_dataloader, torque_model, optimizer)
        torque_val_loop(val_dataloader, torque_model, file_torque)
    print('Done!')


def main():

    torque_model_train(
        class_torque=TorqueModel,
        file_torque='dicts/dict_torque_new.pt',
        learning_rate=3e-4,
        batch_size=256,
        epoches=300
    )


if __name__ == '__main__':

    main()
