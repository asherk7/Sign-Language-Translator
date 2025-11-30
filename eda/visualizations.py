import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os

def get_save_dir():
    """Get the save directory relative to this file."""
    return os.path.join(os.path.dirname(__file__), 'images')

def plot_image(dataloader, classes):
    """
    Displays the first image in a batch with its label.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing image batches.
        classes (list): List of class names.
    """
    image_batch, label_batch = next(iter(dataloader))
    image, label = image_batch[0], label_batch[0]

    plt.imshow(image.permute(1, 2, 0)) # Rearrange image dimensions to suit matplotlib, (rgb, height, width) -> (height, width, rgb)
    plt.title(classes[label])
    plt.axis(False)
    plt.show() 

def plot_training_curves(results):
    """
    Plots training and validation accuracy and loss curves.

    Args:
        results (dict): Dictionary containing training and validation metrics:
                        keys = ["train_acc", "val_acc", "train_loss", "val_loss"]
    """
    train_acc = results["train_acc"]
    val_acc = results["val_acc"]
    train_loss = results["train_loss"]
    val_loss = results["val_loss"]
    epochs = range(len(train_acc))

    plt.figure(figsize=(15, 7))

    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train_accuracy")
    plt.plot(epochs, val_acc, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    save_dir = get_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'training_graph.png')
    plt.savefig(save_path)
    print(f"Training curves saved to {save_path}")

def get_matrix(y_pred, y_true, classes):
    """
    Plots a confusion matrix heatmap.

    Args:
        y_pred (list or array): Model predictions.
        y_true (list or array): True labels.
        classes (list): List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes, yticklabels=classes,
                cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    save_dir = get_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")


def per_class_accuracy(y_pred, y_true, classes):
    """
    Compute and plot per-class accuracy to identify strong/weak classes.
    
    Returns:
        dict: Per-class accuracy scores sorted by performance.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    class_acc = {}
    for i, cls in enumerate(classes):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc[cls] = (y_pred[mask] == y_true[mask]).mean()
        else:
            class_acc[cls] = 0.0
    
    sorted_acc = dict(sorted(class_acc.items(), key=lambda x: x[1]))
    
    plt.figure(figsize=(14, 8))
    colors = ['#ff6b6b' if v < 0.8 else '#4ecdc4' if v > 0.95 else '#95a5a6' for v in sorted_acc.values()]
    plt.barh(list(sorted_acc.keys()), list(sorted_acc.values()), color=colors)
    plt.xlabel('Accuracy')
    plt.title('Per-Class Accuracy (Red < 80%, Green > 95%)')
    plt.xlim(0, 1)
    plt.axvline(x=0.9, color='black', linestyle='--', alpha=0.5, label='90% threshold')
    plt.tight_layout()
    
    save_dir = get_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'per_class_accuracy.png')
    plt.savefig(save_path)
    print(f"Per-class accuracy saved to {save_path}")
    
    print("\nTop 5 Best Classes:")
    for cls, acc in list(reversed(list(sorted_acc.items())))[:5]:
        print(f"  {cls}: {acc:.2%}")
    
    print("\nTop 5 Worst Classes:")
    for cls, acc in list(sorted_acc.items())[:5]:
        print(f"  {cls}: {acc:.2%}")
    
    return sorted_acc


def most_confused_pairs(y_pred, y_true, classes, top_n=10):
    """
    Find the most commonly confused class pairs.
    
    Returns:
        list: Top confused pairs with counts.
    """
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    
    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                confused_pairs.append((classes[i], classes[j], cm[i, j]))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nTop {top_n} Most Confused Pairs:")
    print("(True → Predicted: Count)")
    for true_cls, pred_cls, count in confused_pairs[:top_n]:
        print(f"  {true_cls} → {pred_cls}: {count}")
    
    if confused_pairs:
        top_pairs = confused_pairs[:top_n]
        labels = [f"{t}→{p}" for t, p, _ in top_pairs]
        counts = [c for _, _, c in top_pairs]
        
        plt.figure(figsize=(12, 6))
        plt.barh(labels[::-1], counts[::-1], color='coral')
        plt.xlabel('Misclassification Count')
        plt.title(f'Top {top_n} Most Confused Class Pairs')
        plt.tight_layout()
        
        save_dir = get_save_dir()
        save_path = os.path.join(save_dir, 'confused_pairs.png')
        plt.savefig(save_path)
        print(f"Confused pairs plot saved to {save_path}")
    
    return confused_pairs[:top_n]


def plot_label_distribution(dataloader, classes):
    """
    Plot the distribution of labels in a dataset.
    
    Args:
        dataloader: DataLoader to analyze.
        classes: List of class names.
    """
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.numpy())
    
    label_counts = np.bincount(all_labels, minlength=len(classes))
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(classes, label_counts, color='steelblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Label Distribution')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, label_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_dir = get_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'label_distribution.png')
    plt.savefig(save_path)
    print(f"Label distribution saved to {save_path}")
    
    return label_counts


def plot_misclassified_images(model, dataloader, classes, device, num_images=16):
    """
    Plot a 4x4 grid of misclassified images with predicted and true labels.
    
    Args:
        model: Trained model.
        dataloader: Test dataloader.
        classes: List of class names.
        device: torch device.
        num_images: Number of misclassified images to show (default 16 for 4x4).
    """
    import torch
    
    model.eval()
    misclassified = []
    
    # ImageNet denormalization values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Find misclassified
            wrong_mask = predicted != labels
            wrong_indices = wrong_mask.nonzero(as_tuple=True)[0]
            
            for idx in wrong_indices:
                if len(misclassified) >= num_images:
                    break
                misclassified.append({
                    'image': images[idx].cpu(),
                    'predicted': predicted[idx].item(),
                    'true': labels[idx].item()
                })
            
            if len(misclassified) >= num_images:
                break
    
    if len(misclassified) == 0:
        print("No misclassified images found!")
        return
    
    # Plot 4x4 grid
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(misclassified):
            item = misclassified[i]
            img = item['image'].permute(1, 2, 0).numpy()
            
            # Denormalize
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            pred_label = classes[item['predicted']]
            true_label = classes[item['true']]
            
            ax.imshow(img)
            ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", 
                        color='red', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle('Misclassified Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_dir = get_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'misclassified_images.png')
    plt.savefig(save_path)
    print(f"Misclassified images saved to {save_path}")


def error_analysis_summary(y_pred, y_true, classes):
    """
    Print a comprehensive error analysis summary.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    total = len(y_true)
    correct = (y_pred == y_true).sum()
    incorrect = total - correct
    
    print("Error analysis summary:")
    print(f"Total samples: {total}")
    print(f"Correct: {correct} ({correct/total:.2%})")
    print(f"Incorrect: {incorrect} ({incorrect/total:.2%})")
    
    perfect_classes = []
    for i, cls in enumerate(classes):
        mask = y_true == i
        if mask.sum() > 0 and (y_pred[mask] == y_true[mask]).all():
            perfect_classes.append(cls)
    
    if perfect_classes:
        print(f"\nPerfect Classes (100% accuracy): {len(perfect_classes)} ===")
        print(f"  {', '.join(perfect_classes[:10])}{'...' if len(perfect_classes) > 10 else ''}")
    
    error_mask = y_pred != y_true
    error_true = y_true[error_mask]
    error_counts = np.bincount(error_true, minlength=len(classes))
    
    print(f"\nClasses Contributing Most Errors:")
    top_error_idx = np.argsort(error_counts)[::-1][:5]
    for idx in top_error_idx:
        if error_counts[idx] > 0:
            print(f"  {classes[idx]}: {error_counts[idx]} errors")


def visualize(results, y_pred, y_true, classes, model=None, test_dataloader=None, device=None):
    """
    Combines training curve and confusion matrix visualizations.

    Args:
        results (dict): Training/validation metrics.
        y_pred (list or array): Model predictions.
        y_true (list or array): Ground truth labels.
        classes (list): Class names.
        model (optional): Trained model for misclassified images plot.
        test_dataloader (optional): Test dataloader for misclassified images.
        device (optional): torch device.
    """
    plot_training_curves(results)
    get_matrix(y_pred, y_true, classes)
    per_class_accuracy(y_pred, y_true, classes)
    most_confused_pairs(y_pred, y_true, classes)
    error_analysis_summary(y_pred, y_true, classes)
    
    # Plot misclassified images if model and dataloader provided
    if model is not None and test_dataloader is not None and device is not None:
        plot_misclassified_images(model, test_dataloader, classes, device)
    