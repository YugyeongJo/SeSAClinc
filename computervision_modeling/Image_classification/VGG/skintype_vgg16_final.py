import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter
from google.colab import files

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data directory and categories
data_dir = '/경로'
categories = ['dry', 'oily', 'normal']
label_dict = {'dry': 0, 'oily': 1, 'normal': 2}

# Hyperparameters
hyperparameters = {
    'batch_size': 32,
    'learning_rate': 0.0005,
    'num_epochs': 150,
    'dropout_rate': 0.4,
    'weight_decay': 1e-4,
}

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Data transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_data():
    all_data = []
    all_labels = []
    for idx, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        for file_name in os.listdir(category_dir):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                all_data.append(os.path.join(category_dir, file_name))
                all_labels.append(idx)
    return all_data, all_labels

all_data, all_labels = load_data()

# Train-test split
train_data, valid_data, train_labels, valid_labels = train_test_split(
    all_data, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def create_weighted_sampler(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {class_idx: total_samples / count for class_idx, count in class_counts.items()}
    weights = [class_weights[label] for label in labels]
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler

train_dataset = CustomDataset(train_data, train_labels, transform=transform_train)
valid_dataset = CustomDataset(valid_data, valid_labels, transform=transform_valid)

train_sampler = create_weighted_sampler(train_labels)
train_loader = DataLoader(
    train_dataset,
    batch_size=hyperparameters['batch_size'],
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=hyperparameters['batch_size'],
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Replace AlexNet with VGG16
class VGG16Model(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Model, self).__init__()
        self.backbone = models.vgg16(pretrained=True)
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(2048, num_classes)
        )

        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.backbone(x)

model = VGG16Model(num_classes=len(categories)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=hyperparameters['learning_rate'],
    weight_decay=hyperparameters['weight_decay'],
    amsgrad=True
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3,
    verbose=True
)

# Training loop
def train(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs):
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    best_model_wts = model.state_dict()
    best_acc = 0.0

    early_stopping = EarlyStopping(patience=7)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        valid_losses.append(val_loss)
        valid_accuracies.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")


        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break


        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return train_losses, valid_losses, train_accuracies, valid_accuracies



def predict_image(model, image_path, transform, categories):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prob, predicted = torch.max(probabilities, 1)

    predicted_class = categories[predicted.item()]
    probability = prob.item() * 100

    print(f'예측된 피부 타입: {predicted_class} (확률: {probability:.2f}%)')

    plt.figure(figsize=(8, 8))
    plt.imshow(Image.open(image_path))
    plt.title(f'predict_result: {predicted_class} ({probability:.2f}%)')
    plt.axis('off')
    plt.show()


def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, valid_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, valid_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

train_losses, valid_losses, train_accuracies, valid_accuracies = train(
    model, criterion, optimizer, scheduler, train_loader, valid_loader,
    hyperparameters['num_epochs']
)

torch.save(model.state_dict(), 'best_vgg16_model.pth')
files.download('best_vgg16_model.pth')


model = VGG16Model(num_classes=len(categories)).to(device)


model.load_state_dict(torch.load('best_vgg16_model.pth'))
model.eval()


plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies)


test_image_path = '/경로'
predict_image(model, test_image_path, transform_valid, categories)


