import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from data_preprocessing import HairRemovalTransform
from dataset_utils import prepare_combined_dataset
from model import ResNetBase
from training_utils import train_and_evaluate
from visualization import visualize_metrics

if __name__ == "__main__":
    # Define paths and hyperparameters
    DATASET_PATHS = {
        "isic2020": "./dataset/isic2020/jpeg",
        "isic2024": "./dataset/isic2024/train-image",
        "synthetic": "./dataset/synthetic/images",
    }
    ROOT_DIR = "./dataset"
    NUM_WORKERS = 8
    BATCH_SIZE = 32
    IMG_SIZE = 224
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data loaders
    train_loader, valid_loader, test_loader = prepare_combined_dataset(
        DATASET_PATHS, ROOT_DIR, BATCH_SIZE, NUM_WORKERS
    )

    # Initialize model, optimizer, and criterion
    model = ResNetBase(n_classes=2, pretrained=True)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = CrossEntropyLoss()

    # Train and evaluate
    train_losses, valid_losses, train_accuracies, valid_accuracies = train_and_evaluate(
        model, train_loader, valid_loader, criterion, optimizer, DEVICE, NUM_EPOCHS
    )

    # Visualize metrics
    visualize_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, NUM_EPOCHS)

    print("Training complete. Best model saved as 'best_model.pth'.")
