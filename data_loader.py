import pandas as pd
from PIL import Image
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


from torchvision import transforms

class MisogynyDataset(Dataset):

    def __init__(self, data, label_map, transform=None):
        self.data = data.reset_index(drop=True)
        self.label_map = label_map

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        label = self.label_map[row["image_label"]]
        caption = row["image_caption"]

        return image, caption, label


class MisogynyDataLoader:
    def __init__(
        self,
        csv_file: str = "data_csv.csv",
        batch_size: int = 16,
        test_size: float = 0.2,
        random_state: int = 42,
        train_transform: Optional[callable] = None,
        test_transform: Optional[callable] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        data = pd.read_csv(csv_file)

        label_map = {
            "kitchen": 0,
            "shopping": 1,
            "working": 2,
            "leadership": 3
        }

        train_df, test_df = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=data["image_label"],
        )

        self.train_dataset = MisogynyDataset(
            data=train_df,
            label_map=label_map,
            transform=train_transform,
        )

        self.test_dataset = MisogynyDataset(
            data=test_df,
            label_map=label_map,
            transform=test_transform,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


if __name__ == "__main__":
    dataloaders = MisogynyDataLoader()
    train_loader = dataloaders.train_loader

    print(len(train_loader))
