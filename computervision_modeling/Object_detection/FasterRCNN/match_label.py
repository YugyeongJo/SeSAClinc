import os
import numpy as np
import cv2
import torch


def calculate_correct_detections(predictions, targets, iou_threshold=0.5):
    """
    Count correct detections in a batch.
    
    Args:
        predictions: List of dicts, each containing 'boxes' (predicted bounding boxes) and 'labels' (predicted labels).
        targets: List of dicts, each containing 'boxes' (ground truth bounding boxes) and 'labels' (ground truth labels).
        iou_threshold: IoU threshold to consider a detection correct.
        
    Returns:
        correct: Number of correct detections.
        total_predictions: Total number of predictions.
    """
    correct = 0
    total_predictions = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']  # Predicted boxes: (N, 4)
        pred_labels = pred['labels']  # Predicted labels: (N,)
        gt_boxes = target['boxes']  # Ground truth boxes: (M, 4)
        gt_labels = target['labels']  # Ground truth labels: (M,)

        # Calculate IoU between all predictions and ground truths
        iou_matrix = calculate_iou_matrix(pred_boxes, gt_boxes)

        # Match predictions with ground truths based on IoU
        matched_gt = set()
        for i, pred_box in enumerate(pred_boxes):
            max_iou, max_iou_idx = torch.max(iou_matrix[i], dim=0)
            if max_iou > iou_threshold and max_iou_idx.item() not in matched_gt:
                # Check label match
                if pred_labels[i] == gt_labels[max_iou_idx]:
                    correct += 1
                matched_gt.add(max_iou_idx.item())

        total_predictions += len(pred_boxes)


    return correct, total_predictions


def calculate_iou_matrix(boxes1, boxes2):
    """
    Compute IoU matrix between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape (N, 4), predicted boxes.
        boxes2: Tensor of shape (M, 4), ground truth boxes.
        
    Returns:
        iou_matrix: Tensor of shape (N, M) with IoU values.
    """
    # Calculate areas of boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersections
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Compute unions
    union_area = area1[:, None] + area2 - inter_area

    # IoU
    iou_matrix = inter_area / union_area
    return iou_matrix