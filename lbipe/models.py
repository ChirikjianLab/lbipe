import torch
from torch import nn


class TorqueModel(nn.Module):
    def __init__(self):
        super(TorqueModel, self).__init__()
        self.j1 = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 12)
        )
        self.j2 = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 12)
        )
        self.j3 = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 12)
        )
        self.j4 = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 12)
        )
        self.tau = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x1 = x[:, 0:4]
        x2 = x[:, 4:8]
        x3 = x[:, 8:12]
        x4 = x[:, 12:16]
        h1 = self.j1(x1)
        h2 = self.j2(x2)
        h3 = self.j3(x3)
        h4 = self.j4(x4)
        h = torch.concat([h1, h2, h3, h4], dim=1)
        y = self.tau(h)

        return y


class AttnModel(nn.Module):
    def __init__(self):
        super(AttnModel, self).__init__()
        self.joint1_feature = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 12)
        )
        self.joint2_feature = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 12)
        )
        self.joint3_feature = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 12)
        )
        self.joint4_feature = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 12)
        )
        self.scorer = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = x[:, 0:4]
        x2 = x[:, 4:8]
        x3 = x[:, 8:12]
        x4 = x[:, 12:16]

        f1 = self.joint1_feature(x1)
        f2 = self.joint2_feature(x2)
        f3 = self.joint3_feature(x3)
        f4 = self.joint4_feature(x4)

        num_data = x.size(dim=0)
        idx1 = torch.tensor([1, 0, 0, 0])
        idx2 = torch.tensor([0, 1, 0, 0])
        idx3 = torch.tensor([0, 0, 1, 0])
        idx4 = torch.tensor([0, 0, 0, 1])
        idx1 = idx1.unsqueeze(0).repeat(num_data, 1)
        idx2 = idx2.unsqueeze(0).repeat(num_data, 1)
        idx3 = idx3.unsqueeze(0).repeat(num_data, 1)
        idx4 = idx4.unsqueeze(0).repeat(num_data, 1)
        h1 = torch.cat((f1, idx1), dim=1)
        h2 = torch.cat((f2, idx2), dim=1)
        h3 = torch.cat((f3, idx3), dim=1)
        h4 = torch.cat((f4, idx4), dim=1)
        s1 = self.scorer(h1)
        s2 = self.scorer(h2)
        s3 = self.scorer(h3)
        s4 = self.scorer(h4)
        s = torch.cat((s1, s2, s3, s4), dim=1)
        w = self.softmax(s)

        return w
