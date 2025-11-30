import torch
import torch.nn as nn
from torchvision import models
from collections import Counter
import numpy as np

class MajorityVoteBaseline:
    """Always predicts the most frequent class from training data."""
    def __init__(self):
        self.majority_class = None
    
    def fit(self, train_dataloader):
        all_labels = []
        for _, labels in train_dataloader:
            all_labels.extend(labels.numpy())
        self.majority_class = Counter(all_labels).most_common(1)[0][0]
    
    def predict(self, dataloader):
        predictions = []
        for X, _ in dataloader:
            predictions.extend([self.majority_class] * X.size(0))
        return predictions


class SimpleCNN(nn.Module):
    """Basic 3-layer CNN for image classification."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_mobilenetv2_raw(num_classes):
    """MobileNetV2 trained from scratch (random weights, all layers trainable)."""
    model = models.mobilenet_v2(weights=None) 
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


class RandomForestBaseline:
    """Random Forest classifier on downsampled flattened pixels."""
    def __init__(self, n_estimators=100):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    
    def fit(self, train_dataloader, downsample_size=32):
        X_train, y_train = [], []
        for images, labels in train_dataloader:
            images = torch.nn.functional.interpolate(images, size=(downsample_size, downsample_size))
            X_train.append(images.view(images.size(0), -1).numpy())
            y_train.extend(labels.numpy())
        X_train = np.vstack(X_train)
        self.model.fit(X_train, y_train)
    
    def predict(self, dataloader, downsample_size=32):
        X_test = []
        for images, _ in dataloader:
            images = torch.nn.functional.interpolate(images, size=(downsample_size, downsample_size))
            X_test.append(images.view(images.size(0), -1).numpy())
        X_test = np.vstack(X_test)
        return self.model.predict(X_test)


def train_baseline(model, train_dl, val_dl, test_dl, device, epochs=5, lr=0.001):
    """Train a PyTorch baseline model using existing pipeline."""
    from pipeline.train import train
    from pipeline.test import test
    from utils import get_metrics
    
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    results = train(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device
    )
    
    y_pred, y_true = test(model=model, test_dataloader=test_dl, device=device)
    get_metrics(y_true, y_pred)
    
    return results


def run_all_baselines(train_dl, val_dl, test_dl, num_classes, device):
    """Run all baseline experiments."""
    from pipeline.test import test
    from utils import get_metrics
    
    results = {}
    
    print("\n" + "="*50)
    print("BASELINE: Majority Vote")
    print("="*50)
    mv = MajorityVoteBaseline()
    mv.fit(train_dl)
    y_true = [y for _, labels in test_dl for y in labels.numpy()]
    y_pred = mv.predict(test_dl)
    results["majority_vote"] = get_metrics(y_true, y_pred)
    
    print("\n" + "="*50)
    print("BASELINE: Simple CNN")
    print("="*50)
    cnn = SimpleCNN(num_classes)
    results["simple_cnn"] = train_baseline(cnn, train_dl, val_dl, test_dl, device, epochs=5)
    
    print("\n" + "="*50)
    print("BASELINE: MobileNetV2 Raw (No Pretraining)")
    print("="*50)
    mobilenet_raw = get_mobilenetv2_raw(num_classes)
    results["mobilenetv2_raw"] = train_baseline(mobilenet_raw, train_dl, val_dl, test_dl, device, epochs=5)

    print("\n" + "="*50)
    print("BASELINE: Random Forest")
    print("="*50)
    rf = RandomForestBaseline(n_estimators=100)
    rf.fit(train_dl, downsample_size=32)
    y_true = [y for _, labels in test_dl for y in labels.numpy()]
    y_pred = rf.predict(test_dl, downsample_size=32)
    results["random_forest"] = get_metrics(y_true, y_pred)
    
    return results


if __name__ == "__main__":
    from eda.data_setup import create_dataloaders, transform_images
    from utils import set_seeds
    
    set_seeds()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ASL_DIRS = [
        '/root/.cache/kagglehub/datasets/kapillondhe/american-sign-language/versions/1/ASL_Dataset/Train',
        '/root/.cache/kagglehub/datasets/kapillondhe/american-sign-language/versions/1/ASL_Dataset/Test',
    ]
    
    train_dl, val_dl, test_dl, classes = create_dataloaders(
        data_dirs=ASL_DIRS,
        train_transform=transform_images(train=True),
        test_transform=transform_images(train=False),
        batch_size=32
    )
    
    run_all_baselines(train_dl, val_dl, test_dl, len(classes), device)
