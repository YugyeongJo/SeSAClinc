import torch
from torchvision.ops import box_iou

def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    mAP (Mean Average Precision) 계산 함수
    Args:
        predictions: 모델의 예측 출력 (list of dict)
        targets: 실제 정답 데이터 (list of dict)
        iou_threshold: IoU 기준 (default=0.5)

    Returns:
        mAP (float): 평균 평균 정밀도
    """
    average_precisions = []

    for pred, target in zip(predictions, targets):
        if len(pred["boxes"]) == 0 or len(target["boxes"]) == 0:
            # 예측 또는 정답이 없을 경우 AP를 0으로 설정
            average_precisions.append(0.0)
            continue

        # IoU 계산
        ious = box_iou(pred["boxes"], target["boxes"])

        # 정렬된 IoU 및 매칭 여부 계산
        iou_max, indices = ious.max(dim=1)
        detections_matched = iou_max >= iou_threshold

        # TP, FP, FN 계산
        true_positives = detections_matched.sum().item()
        false_positives = len(pred["boxes"]) - true_positives
        false_negatives = len(target["boxes"]) - true_positives

        # Precision, Recall 계산
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

        # Average Precision (AP) 계산
        if precision + recall > 0:
            average_precision = 2 * (precision * recall) / (precision + recall)
        else:
            average_precision = 0.0

        average_precisions.append(average_precision)

    # mAP 계산
    mAP = sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    return mAP