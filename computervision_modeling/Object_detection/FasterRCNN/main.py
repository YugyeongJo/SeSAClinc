# main.py
import torch
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from fasterrcnn_model import FRCNNObjectDetector
from train_evaluation import train, evaluate, save_model
from data import get_data_loader
from torchvision import transforms

def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, test_accuracies, save_path):
    """학습 과정의 loss와 accuracy를 그래프로 저장"""
    epochs = range(1, len(train_losses) + 1)

    # Loss 그래프
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, valid_losses, label='Validation Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train mAP', marker='o')
    plt.plot(epochs, valid_accuracies, label='Validation mAP', marker='o')
    plt.plot(epochs, test_accuracies, label='Test mAP', marker='o')
    plt.title('mAP Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")
    plt.close()


def main():
    # 경로 설정
    root_dir = '../ddataset/'
    coco_annotations_dir = '../ddataset/coco_annotations/'

    # 카테고리 리스트
    categories = ['acne', 'pigmentation', 'psoriasis']

    # 모델 설정
    num_classes = 3  # 여드름, 색소침착, 건선
    model = FRCNNObjectDetector(num_classes=num_classes)
    model.to(device)

    # Optimizer 설정
    optimizer = Adam(model.parameters(), lr=lr)

    # 저장할 디렉토리 설정
    save_dir = './saved_models'
    os.makedirs(save_dir, exist_ok=True)  # 모델 저장 디렉토리 생성
    plots_dir = './saved_plots'
    os.makedirs(plots_dir, exist_ok=True)  # 그래프 저장 디렉토리 생성

    # Metrics 저장용 리스트
    train_losses, valid_losses, test_losses = [], [], []
    train_accuracies, valid_accuracies, test_accuracies = [], [], []

    # 학습 진행
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        epoch_train_loss, epoch_train_map = 0.0, 0.0
        epoch_valid_map, epoch_test_map = 0.0, 0.0

        # 각 카테고리마다 데이터를 로드하여 학습
        for category in categories:
            print(f"Processing {category}...")

            # 어노테이션 경로
            train_annotation = os.path.join(coco_annotations_dir, f'{category}_train_annotations.json')
            valid_annotation = os.path.join(coco_annotations_dir, f'{category}_valid_annotations.json')
            test_annotation = os.path.join(coco_annotations_dir, f'{category}_test_annotations.json')

            # 이미지 디렉토리 경로 설정
            train_images_dir = os.path.join(root_dir, category, 'train', 'images').replace('/', '\\')
            valid_images_dir = os.path.join(root_dir, category, 'valid', 'images').replace('/', '\\')
            test_images_dir = os.path.join(root_dir, category, 'test', 'images').replace('/', '\\')

            # 데이터 로더 생성
            transform = transforms.Compose([transforms.ToTensor()])
            train_loader = get_data_loader(train_images_dir, train_annotation, batch_size=batch_size, transforms=transform)
            valid_loader = get_data_loader(valid_images_dir, valid_annotation, batch_size=batch_size, transforms=transform)
            test_loader = get_data_loader(test_images_dir, test_annotation, batch_size=batch_size, transforms=transform)

            # Train
            train_loss, train_map = train(model, train_loader, optimizer, device)
            epoch_train_loss += train_loss
            epoch_train_map += train_map

            # Validation
            valid_map = evaluate(model, valid_loader, device)
            epoch_valid_map += valid_map

            # Test
            test_map = evaluate(model, test_loader, device)
            epoch_test_map += test_map

        # 평균 Loss 및 mAP 계산
        train_losses.append(epoch_train_loss / len(categories))
        train_accuracies.append(epoch_train_map / len(categories))
        valid_accuracies.append(epoch_valid_map / len(categories))
        test_accuracies.append(epoch_test_map / len(categories))

        # Epoch 결과 출력
        print(f"Train Loss: {train_losses[-1]:.4f}, Train mAP: {train_accuracies[-1]:.4f}")
        print(f"Valid mAP: {valid_accuracies[-1]:.4f}")
        print(f"Test mAP: {test_accuracies[-1]:.4f}")

        # 모델 저장
        model_save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
        save_model(model, model_save_path)

    # 학습 과정 그래프 저장
    plot_save_path = os.path.join(plots_dir, 'metrics_plot.png')
    plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, test_accuracies, plot_save_path)

    print("Training and evaluation completed!")


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    num_epochs = 1  # 학습 횟수 설정
    batch_size = 8
    lr = 1e-4
    main()
