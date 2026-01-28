from medmnist import PneumoniaMNIST, PathMNIST, DermaMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

DATASETS = {
    "Chest (Pneumonia)": {
        "class": PneumoniaMNIST,
        "channels": 1
    },
    "Pathology (Tumor)": {
        "class": PathMNIST,
        "channels": 3
    },
    "Skin (Dermatology)": {
        "class": DermaMNIST,
        "channels": 3
    }
}

def get_dataloaders(dataset_name, num_clients=3, batch_size=4):
    transform = transforms.ToTensor()

    DatasetClass = DATASETS[dataset_name]["class"]

    dataset = DatasetClass(
        split='train',
        transform=transform,
        download=True
    )

    data_len = len(dataset)
    indices = np.random.permutation(data_len)
    split = np.array_split(indices, num_clients)

    loaders = []
    for i in range(num_clients):
        subset = Subset(dataset, split[i])
        loaders.append(
            DataLoader(subset, batch_size=batch_size, shuffle=True)
        )

    return loaders, DATASETS[dataset_name]["channels"]