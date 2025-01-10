import time
import torch
from sklearn.metrics import roc_curve, auc

def train_and_evaluate(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs):
    """
    Train and evaluate the model over multiple epochs.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        valid_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run training on (CPU or GPU).
        num_epochs: Number of epochs to train.

    Returns:
        tuple: Training and validation losses and accuracies.
    """
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    best_valid_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        start_time = time.time()

        # Training
        train_loss, train_acc = model.train_one_epoch(train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Training: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")

        # Validation
        valid_loss, valid_acc = model.validate_one_epoch(valid_loader, criterion, device)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        print(f"Validation: Loss = {valid_loss:.4f}, Accuracy = {valid_acc:.4f}")

        # Save the best model
        if valid_acc > best_valid_acc:
            print(f"Validation accuracy improved from {best_valid_acc:.4f} to {valid_acc:.4f}. Saving model...")
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), "best_model.pth")

        epoch_time = time.time() - start_time
        print(f"Time for epoch {epoch + 1}: {epoch_time:.2f} seconds")

    return train_losses, valid_losses, train_accuracies, valid_accuracies

def calculate_partial_auc(y_true, y_scores, fpr_threshold=0.8):
    """
    Calculate the partial AUC (pAUC) for TPR >= threshold.

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_scores (array-like): Predicted probabilities for the positive class.
        fpr_threshold (float): Maximum FPR threshold for pAUC calculation.

    Returns:
        float: Partial AUC value above the specified FPR threshold.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fpr_partial = fpr[fpr <= fpr_threshold]
    tpr_partial = tpr[:len(fpr_partial)]
    return auc(fpr_partial, tpr_partial) if len(tpr_partial) > 1 else 0.0
