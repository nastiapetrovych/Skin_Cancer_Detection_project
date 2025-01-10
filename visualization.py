import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def visualize_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, num_epochs):
    """
    Visualize training and validation metrics over epochs.

    Args:
        train_losses (list): Training loss for each epoch.
        valid_losses (list): Validation loss for each epoch.
        train_accuracies (list): Training accuracy for each epoch.
        valid_accuracies (list): Validation accuracy for each epoch.
        num_epochs (int): Number of epochs.
    """
    epochs = range(1, num_epochs + 1)

    # Plot Loss
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, valid_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, valid_accuracies, label='Validation Accuracy', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_validation_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_with_threshold(y_true, y_scores, threshold=0.8):
    """
    Plot the ROC curve and highlight the partial AUC (pAUC) above a given FPR threshold.

    Args:
        y_true (list): True labels.
        y_scores (list): Predicted probabilities for the positive class.
        threshold (float): FPR threshold for pAUC calculation.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fpr_partial = fpr[fpr <= threshold]
    tpr_partial = tpr[:len(fpr_partial)]
    pauc = auc(fpr_partial, tpr_partial)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.fill_between(fpr_partial, 0, tpr_partial, color='blue', alpha=0.3,
                     label=f'pAUC (FPR <= {threshold:.2f}) = {pauc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Partial AUC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_with_pauc.png', dpi=300, bbox_inches='tight')
    plt.show()
