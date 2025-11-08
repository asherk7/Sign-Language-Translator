import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, Dataset
import torch

# Use half the CPU cores or 0 if none are available
NUM_WORKERS = (os.cpu_count() // 2) or 0  

class TransformSubset(Dataset):
    """Wrapper to apply different transforms to a subset of data."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        # Expose dataset attribute for compatibility
        self.dataset = subset.dataset if hasattr(subset, 'dataset') else subset
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)
    
# Create ImageFolder with custom find_classes to filter 'del' and normalize case
class CustomImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        # Get all directories with their original names
        entries = [(entry.name, entry.name.lower()) for entry in os.scandir(directory) if entry.is_dir()]
        # Filter out 'del' (case-insensitive)
        entries = [(original, lower) for original, lower in entries if lower not in ['del']]
        
        if not entries:
            raise FileNotFoundError(f"Couldn't find any class folders in {directory}.")
        
        # Group by lowercase name (to handle Space/space)
        from collections import defaultdict
        lower_to_original = defaultdict(list)
        for original, lower in entries:
            lower_to_original[lower].append(original)
        
        # Use the first occurrence of each lowercase class
        classes = []
        original_names = {}  # Map lowercase -> original folder name
        for lower in sorted(lower_to_original.keys()):
            classes.append(lower)
            original_names[lower] = lower_to_original[lower][0]  # Use first occurrence
        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        # Store mapping for later use
        self.original_names = original_names
        
        return classes, class_to_idx
    
    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None, allow_empty=False):
        # Override to use original folder names
        instances = []
        directory = os.path.expanduser(directory)
        
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            # Use the original folder name
            original_folder = self.original_names.get(target_class, target_class)
            target_dir = os.path.join(directory, original_folder)
            
            if not os.path.isdir(target_dir):
                continue
                
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file is not None:
                        if is_valid_file(path):
                            instances.append((path, class_index))
                    elif extensions is not None:
                        if path.lower().endswith(extensions):
                            instances.append((path, class_index))
                    else:
                        instances.append((path, class_index))
        
        if not allow_empty and len(instances) == 0:
            raise FileNotFoundError(f"Found no valid files in {directory}")
        
        return instances

def create_dataloaders(data_dirs: list, 
                       train_transform: transforms.Compose, 
                       test_transform: transforms.Compose, 
                       batch_size: int, 
                       num_workers: int = NUM_WORKERS):
    """
    Creates DataLoaders for training, validation, and testing datasets.
    Combines multiple dataset directories into one unified dataset.

    Args:
        data_dirs (list): List of paths to all data directories to combine.
        train_transform (transforms.Compose): Transformations for training images.
        test_transform (transforms.Compose): Transformations for validation/testing images.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader, class_names)
    """

    # Load all datasets with case-insensitive class names
    all_datasets = []
    for data_dir in data_dirs:
        print(f"Loading data from: {data_dir}")
        dataset = CustomImageFolder(data_dir, transform=None)
        all_datasets.append(dataset)
    
    # Combine all datasets into one
    combined_data = ConcatDataset(all_datasets)
    classes = all_datasets[0].classes
    
    print(f"Total images: {len(combined_data)}")
    print(f"Classes: {classes}")

    # Split into train (80%), val (10%), test (10%)
    total_size = len(combined_data)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_subset, val_subset, test_subset = random_split(
        combined_data, 
        [train_size, val_size, test_size]
    )

    # Apply transforms to each split
    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, test_transform)
    test_dataset = TransformSubset(test_subset, test_transform)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    use_pin_memory = torch.cuda.is_available()
    
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=use_pin_memory)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=use_pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=use_pin_memory)

    return train_dataloader, val_dataloader, test_dataloader, classes

def transform_images(train=True):
    """
    Applies image transformations for training or testing datasets.

    Args:
        train (bool): Whether to return the training transforms.

    Returns:
        torchvision.transforms.Compose: Transform pipeline.
    """
    IMG_SIZE = 224 

    if train:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1
            ),
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    return transform

if __name__ == "__main__":
    raise NotImplementedError("Set up paths and transforms before calling create_dataloaders().")