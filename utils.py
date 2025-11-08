import torch
from torchinfo import summary
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

def set_seeds(seed=42):
    """
    Set seed for reproducibility across CPU and GPU.
    """
    torch.manual_seed(seed)

def get_metrics(y_true, y_pred):
    """
    Compute and print classification metrics.

    Args:
        y_true (list or array): Ground truth labels.
        y_pred (list or array): Predicted labels.

    Returns:
        dict: Contains accuracy and full classification report.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'report': classification_report(y_true, y_pred), # Contains f1-score, precision, and recall
    }

    #print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(metrics['report'])

    return metrics

def model_summary(model):
    """
    Print a summary of the model using torchinfo.

    Args:
        model (nn.Module): The model to summarize.

    Returns:
        summary object: Torchinfo summary.
    """
    summary(model=model,
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    return summary

def get_class_distribution(dataset, data):
    """
    Get the distribution of classes in a subset of a dataset.

    Args:
        dataset: Subset (like from random_split).
        data: Full dataset (e.g., ImageFolder) to access `.targets`.

    Returns:
        Counter: Class label counts.
    """
    labels = [data.targets[idx] for idx in dataset.indices]
    return Counter(labels)
