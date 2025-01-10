import torch
import torch.nn as nn
import timm

class ResNetBase(nn.Module):
    """
    ResNet-based model for binary classification.
    """
    def __init__(self, n_classes, pretrained=False, model_path=None):
        super(ResNetBase, self).__init__()
        # Initialize the ResNet model
        self.model = timm.create_model("resnet50", pretrained=pretrained)

        if pretrained and model_path:
            self.model.load_state_dict(torch.load(model_path))

        # Replace the classification head
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

    def forward(self, x):
        return self.model(x)

    def train_one_epoch(self, train_loader, criterion, optimizer, device):
        self.train()
        self.to(device)
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += (output.argmax(dim=1) == target).float().mean().item()

        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = epoch_accuracy / len(train_loader)
        return avg_loss, avg_accuracy

    def validate_one_epoch(self, valid_loader, criterion, device):
        self.eval()
        self.to(device)
        valid_loss = 0.0
        valid_accuracy = 0.0

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                loss = criterion(output, target)

                valid_loss += loss.item()
                valid_accuracy += (output.argmax(dim=1) == target).float().mean().item()

        avg_loss = valid_loss / len(valid_loader)
        avg_accuracy = valid_accuracy / len(valid_loader)
        return avg_loss, avg_accuracy
