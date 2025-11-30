import torch
import torch.nn as nn
from torchvision import models

ABLATION_CONFIGS = {
    # Ablation 1: No data augmentation
    "no_augmentation": {
        "use_augmentation": False,
        "learning_rate": 0.0005,
        "batch_size": 64,
    },
    
    # Ablation 2: No learning rate scheduler
    "no_scheduler": {
        "use_scheduler": False,
        "learning_rate": 0.0005,
        "batch_size": 64,
    },
    
    # Ablation 3: No label smoothing
    "no_label_smoothing": {
        "label_smoothing": 0.0,
        "learning_rate": 0.0005,
        "batch_size": 64,
    },
    
    # Ablation 4: Smaller batch size
    "small_batch": {
        "batch_size": 16,
        "learning_rate": 0.0005,
    },
    
    # Ablation 5: No weight decay
    "no_weight_decay": {
        "weight_decay": 0.0,
        "learning_rate": 0.0005,
        "batch_size": 64,
    },
    
    # Ablation 6: Different optimizer (SGD instead of Adam)
    "sgd_optimizer": {
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "batch_size": 64,
    },
    
    # Ablation 7: Fewer frozen layers
    "partial_freeze": {
        "freeze_layers": 10,  # Only freeze first 10 layers
        "learning_rate": 0.0001,
        "batch_size": 64,
    },
}


def get_ablation_model(config_name, num_classes):
    """Get model configured for specific ablation study."""
    config = ABLATION_CONFIGS.get(config_name, {})
    
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Handle freezing
    if "freeze_layers" in config:
        layers_to_freeze = config["freeze_layers"]
        for i, (name, param) in enumerate(model.named_parameters()):
            if i < layers_to_freeze:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model, config


def run_ablation(config_name, train_dl, val_dl, test_dl, num_classes, device):
    """Run a single ablation experiment."""
    from pipeline.train import train
    from pipeline.test import test
    from utils import get_metrics
    
    model, config = get_ablation_model(config_name, num_classes)
    model = model.to(device)
    
    print(f"\nRunning ablation: {config_name}")
    print(f"Config: {config}")
    
    lr = config.get("learning_rate", 0.0005)
    label_smoothing = config.get("label_smoothing", 0.1)
    weight_decay = config.get("weight_decay", 0.01)
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    if config.get("optimizer") == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, momentum=config.get("momentum", 0.9)
        )
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
    
    if config.get("use_scheduler", True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    results = train(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=5,
        device=device
    )
    
    y_pred, y_true = test(model=model, test_dataloader=test_dl, device=device)
    metrics = get_metrics(y_true, y_pred)
    
    return {"config": config_name, "train_results": results, "test_metrics": metrics}


def run_all_ablations(train_dl, val_dl, test_dl, num_classes, device):
    """Run all ablation experiments."""
    all_results = {}
    for config_name in ABLATION_CONFIGS:
        all_results[config_name] = run_ablation(config_name, train_dl, val_dl, test_dl, num_classes, device)
    return all_results


if __name__ == "__main__":
    print("Available ablation configurations:")
    for name, config in ABLATION_CONFIGS.items():
        print(f"  - {name}: {config}")
