import itertools
import random

import torch as th
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


def mod_div(a, b, m):
    for c in range(m):
        if (b * c) % m == a:
            return c
    return -1


def make_data(base):
    return [[a, b, mod_div(a, b, base)]
            for a, b in itertools.product(range(base), range(1, base))]


class SimpleDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def make_dataset(base, train_frac=0.8, batch_size=32):
    data = make_data(base)
    random.shuffle(data)
    split = int(train_frac*len(data))

    data = th.tensor(data)
    op = th.full((data.shape[0],), base)
    eq = th.full((data.shape[0],), base+1)
    data = th.vstack([data[:, 0], op, data[:, 1], eq, data[:, 2]]).T

    train_data = SimpleDataset(data[:split])
    test_data = SimpleDataset(data[split:])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    base = 97
    data = make_data(base)
    print(data)
    print(len(data))
