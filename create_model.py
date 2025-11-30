import torch
import sys
import os

# Add project root to sys.path to allow importing local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eda.data_setup import create_dataloaders, transform_images
from utils import set_seeds, model_summary, get_metrics
from eda.visualizations import visualize, plot_label_distribution
from pipeline.train import train
from pipeline.test import test
from torchvision import models

# Hyperparameters and configurations
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1

ASL_DIR0 = '/root/.cache/kagglehub/datasets/kapillondhe/american-sign-language/versions/1/ASL_Dataset/Train'
ASL_DIR1 = '/root/.cache/kagglehub/datasets/kapillondhe/american-sign-language/versions/1/ASL_Dataset/Test'
ASL_DIR2 = '/root/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train'
NUM_EPOCHS = 2
BATCH_SIZE = 64  

# Fine tuning
FT_LEARNING_RATE = 0.00005
FT_NUM_EPOCHS = 3
FT_LABEL_SMOOTHING = 0

def main():
    # Auto-detect best available device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"Using device: {device}")
    else:
        device = "cpu"
        print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    set_seeds()

    # Create transforms (image augmentations)
    train_transformer = transform_images(train=True)
    test_transformer = transform_images(train=False)

    # Create DataLoaders
    train_dataloader, val_dataloader, test_dataloader, classes = create_dataloaders(
        data_dirs=[ASL_DIR0, ASL_DIR1, ASL_DIR2],  # Combine all directories
        train_transform=train_transformer,
        test_transform=test_transformer,
        batch_size=BATCH_SIZE
    )

    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")

    # Initialize the model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    model = model.to(device)

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final classifier layer
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    
    model.classifier[1] = model.classifier[1].to(device)

    model_summary(model)

    # Define loss function, optimizer, and scheduler
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2
    )

    # Train the model
    results_main = train(model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    epochs=NUM_EPOCHS,
                    device=device)

    # Fine-tuning 
    print("Starting fine-tuning")
    for param in model.features[-5:].parameters():
        param.requires_grad = True

    # Define new loss function, optimizer, and scheduler for fine-tuning
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=FT_LABEL_SMOOTHING)

    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=FT_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    results_finetune = train(model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    epochs=FT_NUM_EPOCHS,
                    device=device)
    
    # Combine results
    results = {
        "train_loss": results_main["train_loss"] + results_finetune["train_loss"],
        "train_acc": results_main["train_acc"] + results_finetune["train_acc"],
        "val_loss": results_main["val_loss"] + results_finetune["val_loss"],
        "val_acc": results_main["val_acc"] + results_finetune["val_acc"],
    }

    # Test the model
    y_pred, y_true = test(model=model, test_dataloader=test_dataloader, device=device)

    # Plot label distribution
    plot_label_distribution(train_dataloader, classes)

    # Visualize results and calculate metrics
    visualize(results, y_pred, y_true, classes, model=model, test_dataloader=test_dataloader, device=device)
    get_metrics(y_true, y_pred)

    # Save the trained model state dictionary (weights)
    save_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'mobilenet_v2_sign_language.pth')
    torch.save(obj=model.state_dict(), f=save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()