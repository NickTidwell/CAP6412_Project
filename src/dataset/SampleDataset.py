from torch.utils.data import  Dataset
1
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = self.data[idx][:, :, :, 1]  
        target_data = self.data[idx][1, 1, 1, :]  
        return input_data, target_data