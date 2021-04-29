from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class DepressionDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        return self.data[index], self.labels[index]
