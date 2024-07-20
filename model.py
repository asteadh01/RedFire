import torch
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from PIL import Image
import tifffile
import pickle
class SentinelDataset(torch.utils.data.Dataset):
    def __init__(self, size, images_path, labels_path):
        self.imgs = sorted(list(Path(images_path).iterdir()))
        self.labels = sorted(list(Path(labels_path).iterdir()))
        self.crop_size = size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if img_path.suffix == '.png':
            img = Image.open(img_path)
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
        else:
            img = torch.from_numpy(tifffile.imread(img_path).astype(np.float32)) / 255.

        labels = Image.open(self.labels[index])
        labels = np.array(labels)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return img, labels

def indices_split(dataset_path):
    ids = {"train": [], "val": [], "test": []}
    for key, data_id in ids.items():
        with open((dataset_path / f"{key}.txt"), "r") as f:
            for line in f:
                data_id.append(int(Path(line).stem))
    return ids["train"], ids["val"], ids["test"]

def get_datasets(size, base_path):
    train_indices, val_indices, test_indices = indices_split(base_path)
    with open(base_path / "class_mapping.json", "r") as f:
        class_mapping = json.load(f)

    dataset = SentinelDataset(
        size,
        base_path / "images",
        base_path / "labels",
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset, class_mapping

def get_dataloaders(size, base_path, batch_size=32, num_workers=0):
    train_dataset, val_dataset, test_dataset, class_mapping = get_datasets(size, base_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, class_mapping

def train_ML(dataloader, params):
    print("Starting training...")
    rf = RandomForestClassifier()
    rnd_gs = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=10, verbose=2, random_state=42, n_jobs=-1)

    for x_train, y_train in dataloader:
        x_train = x_train.numpy()
        y_train = y_train.numpy()

        B, H, W, C = x_train.shape
        x_train = x_train.reshape(B, H * W, C)
        y_train = y_train.reshape(B, H * W)

        x_train = x_train.reshape(B * H * W, C)
        y_train = y_train.reshape(B * H * W)

        rnd_gs.fit(x_train, y_train)
    print("##########################################")
    return rnd_gs.best_estimator_

model_params = {
    'max_depth': [20, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 4],
    'n_estimators': [100, 150],
    'class_weight': ["balanced_subsample"],
    'criterion': ["entropy", "gini"],
}

if __name__ == "__main__":
    task = 'semantic_segmentation'
    size = 224
    base_pth = Path(f'datasets/preincendio_biobio_sentinel2/train/{task}')

    batch_size = 12
    num_workers = 0

    train_loader, val_loader, test_loader, class_mapping = get_dataloaders(size, base_pth, batch_size=batch_size, num_workers=num_workers)

    model = train_ML(train_loader, model_params)

    # Guardar el modelo entrenado
    model_save_path = "trained_model.pkl"
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)