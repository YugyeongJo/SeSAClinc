# train_evaluation.py
import torch
from map_utils import calculate_map

# Train 함수
# Train 함수
def train(model, train_loader, optimizer, device):
    running_loss = []

    all_outputs = []  # 모든 배치의 모델 예측을 저장
    all_targets = []  # 모든 배치의 정답 데이터를 저장

    for data in train_loader:
        model.train()
        images, targets = data  # images와 targets을 분리
        
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Forward pass (Loss 계산)
        loss_dict = model(images, targets)
        print(f"image", {images})
        print(f"target", {targets})
        loss_classifier_value = loss_dict['loss_classifier']
        losses_float = loss_classifier_value.item()

        # Backward pass
        loss_classifier_value.backward()
        optimizer.step()

        running_loss.append(losses_float)

        # 정확도 및 mAP 계산을 위한 모델 평가
        model.eval()
        with torch.no_grad():
            outputs = model(images)  # 예측 수행
            all_outputs.extend(outputs)
            all_targets.extend(targets)

        print(f"TRAIN 배치당 LOSS: {running_loss[-1]}")

    # 훈련 mAP 계산
    train_mAP = calculate_map(all_outputs, all_targets, iou_threshold=0.5)

    # 평균 손실 반환 (배치당 평균 손실)
    return sum(running_loss) / len(train_loader), train_mAP

# Evaluate 함수
def evaluate(model, data_loader, device):
    model.eval()
    all_outputs = []  # 모든 배치의 모델 예측을 저장
    all_targets = []  # 모든 배치의 정답 데이터를 저장

    with torch.no_grad():
        for data in data_loader:
            images, targets = data
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)  # 모델 예측
            all_outputs.extend(outputs)
            all_targets.extend(targets)

    # mAP 계산
    mAP = calculate_map(all_outputs, all_targets, iou_threshold=0.5)
    return mAP


# 모델 저장 함수
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)  # 모델의 가중치를 저장
    print(f"Model saved to {save_path}")
