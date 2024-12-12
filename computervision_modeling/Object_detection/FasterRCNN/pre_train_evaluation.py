# train_evaluation.py
import torch
import match_label as match
from computervision_modeling.model.matrix_map import calculate_map

# Train 함수
def train(model, train_loader, optimizer, device):

    running_loss = []
    correct_predictions = 0
    total_predictions = 0

    for data in train_loader:
        model.train()
        images, targets = data  # images와 targets을 분리
        # print(f"images : ", images)
        # print(f"targets : ", targets)
        
        images = images.to(device)  # images[0]을 device로 이동시켜야 할 경우
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        loss_classifier_value = loss_dict['loss_classifier']
        losses_float = loss_classifier_value.item()

        # Backward pass
        loss_classifier_value.backward()
        optimizer.step()

        running_loss.append(losses_float)

        # 정확도 계산 (예시로 분류 문제로 가정)
        model.eval()
        with torch.no_grad():
            
            #길이를 맞춰주기 위해서 targets만큼의 수를 만듦
            #각 이미지별로 수행되어야 함
            
            #outputs의 예측
            #outputs의 예측이 per target과 일치하여야 함.
            outputs = model(images)
            #print(outputs)
            
            correct, total_ = match.calculate_correct_detections(outputs, targets, iou_threshold=0.5)
            correct_predictions += correct
            total_predictions += total_
            
            print(f"TRAIN 배치당 LOSS {running_loss[-1]}")

    # 훈련 정확도 계산
    #train_loss, train_acc
    train_accuracy = correct_predictions/total_predictions
    return running_loss[-1] / len(train_loader), train_accuracy


# Evaluate 함수
def evaluate(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for data in data_loader:
            images, targets = data  # images와 targets을 분리
            
            images = images.to(device)  # images[0]을 device로 이동시켜야 할 경우
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#--------------------------------------------------------
# Forward pass
            #길이를 맞춰주기 위해서 targets만큼의 수를 만듦
            #각 이미지별로 수행되어야 함
            
            #outputs의 예측
            #outputs의 예측이 per target과 일치하여야 함.
            outputs = model(images)
            #print(outputs)
            
            correct, total_ = match.calculate_correct_detections(outputs, targets, iou_threshold=0.5)
            correct_predictions += correct
            total_predictions += total_
            
#--------------------------------------------------------
    # 훈련 정확도 계산
    valid_accuracy = correct_predictions/total_predictions
    return valid_accuracy


# 모델 저장 함수
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)  # 모델의 가중치를 저장
    print(f"Model saved to {save_path}")
