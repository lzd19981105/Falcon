import os
import re
import json
import string
import math
import argparse
import numpy as np
from shapely.geometry import Polygon

# Convert a flat list of 8 points into a Polygon object
def process_flat_obb(a):
    if len(a) != 8:  # Check if the input contains exactly 8 points
        print('Error: OBB points number must be 8.')
        return False
    else:
        obb = [(a[0], a[1]), (a[2], a[3]),
               (a[4], a[5]), (a[6], a[7])]
        poly = Polygon(obb)  # Create a Shapely polygon
        if poly.is_valid:  # Check if the polygon is valid
            return poly
        else:
            return False

# Parse and format bounding box data based on input type
def format_preprocess(data, is_box=False, is_obb=False, isGT=False):
    multiplyer = 1  # Multiplier for scaling coordinates (default 1)
    if is_box:  # If processing bounding box data
        if '<quad>' in data or '<box>' in data:
            if is_obb:  # Extract oriented bounding boxes (OBB)
                boxes = re.findall(r'<quad>(.*?)</quad>', data)
                boxes = [[float(num) * multiplyer for num in re.findall(r'\d+', box)] for box in boxes]
            else:  # Extract horizontal bounding boxes (HBB)
                boxes = re.findall(r'<box>(.*?)</box>', data)
                boxes = [[float(num) * multiplyer for num in re.findall(r'\d+', box)] for box in boxes]
        else:  # Return empty if no relevant tags are found
            return []
    else:  # If the input is already a list of coordinates
        if isinstance(data, list):
            boxes = [[float(a) for a in x] for x in data]
        else:
            return []
    return boxes

# Calculate IoU (Intersection over Union) for two OBBs
def calculate_iou_obb(pred_box, gt_box):
    poly1 = process_flat_obb(pred_box)  # Convert predicted box to polygon
    poly2 = process_flat_obb(gt_box)  # Convert ground truth box to polygon

    if not poly1:  # Invalid predicted box
        return 0.0
    if not poly2:  # Invalid ground truth box
        print('Error in ground truth box:', gt_box)
        return 0.0
    
    if not poly1.intersects(poly2):  # No intersection between polygons
        return 0.0
    
    # Calculate areas and IoU
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - intersection_area
    iou = intersection_area / union_area
    return iou

# Calculate IoU for two horizontal bounding boxes (HBB)
def calculate_iou_hbb(box1, box2):
    try:
        # Find the intersection coordinates
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])
    
        # Calculate areas
        inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area + 0.1
    
        # Compute IoU
        iou = inter_area / union_area
    except:
        iou = 1  # Default to 1 in case of calculation errors
    return iou

# Compute average precision (AP) using precision-recall curve
def calculate_ap(precision, recall):
    recall = np.concatenate(([0.0], recall, [1.0]))  # Add endpoints to recall
    precision = np.concatenate(([0.0], precision, [0.0]))  # Add endpoints to precision

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute AP as the area under the precision-recall curve
    indices = np.where(recall[1:] != recall[:-1])[0]  # Points where recall changes
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap

# Evaluate a single set of predicted and ground truth boxes
def evaluate_one(gt_boxes, pred_boxes, iou_threshold=0.1, pred_format=False, gt_format=False, is_obb=False):
    pred_boxes = format_preprocess(pred_boxes, pred_format, is_obb)
    gt_boxes = format_preprocess(gt_boxes, gt_format, is_obb, isGT=True)
    
    tp = np.zeros(len(pred_boxes))  # True positives
    fp = np.zeros(len(pred_boxes))  # False positives
    ious = np.zeros(len(pred_boxes))  # IoU values
    detected_gt = []  # Track matched ground truth boxes

    for i, pred_box in enumerate(pred_boxes):
        match_found = False

        for j, gt_box in enumerate(gt_boxes):
            if is_obb:  # Calculate IoU for OBB
                iou = calculate_iou_obb(pred_box, gt_box)
            else:  # Calculate IoU for HBB
                iou = calculate_iou_hbb(pred_box, gt_box)
            if j not in detected_gt and iou >= iou_threshold:  # Match found
                tp[i] = 1
                ious[i] = iou
                detected_gt.append(j)
                match_found = True
                break

        if not match_found:  # No match found, mark as false positive
            fp[i] = 1

    fn = len(gt_boxes) - len(detected_gt)  # False negatives
    return tp, fp, ious, fn, len(pred_boxes), len(gt_boxes)

# Evaluate multiple detection results and compute overall metrics
def calculate_eval_matrix(data, iou_threshold=0.1, pred_format=False, gt_format=False, is_obb=False):
    total_tp, total_fp, total_ious = np.array([]), np.array([]), np.array([])
    total_fn, total_pres, total_gts = 0, 0, 0

    for a in data:  # Process each sample
        tp, fp, iou, fn, pres, gts = evaluate_one(a['gt'], a['answer'], iou_threshold, pred_format, gt_format, is_obb)
        total_tp = np.concatenate((total_tp, tp))
        total_fp = np.concatenate((total_fp, fp))
        total_ious = np.concatenate((total_ious, iou))
        total_fn += fn
        total_pres += pres
        total_gts += gts

    sorted_indices = np.argsort(total_ious)  # Sort by IoU values
    total_tp = np.array(total_tp)[sorted_indices]
    total_fp = np.array(total_fp)[sorted_indices]

    all_tp = np.cumsum(total_tp)  # Cumulative true positives
    all_fp = np.cumsum(total_fp)  # Cumulative false positives

    recall = all_tp / float(total_gts) if total_gts > 0 else np.zeros(len(all_tp))
    precision = all_tp / list(range(1, total_pres + 1)) if total_pres > 0 else np.zeros(len(all_tp))

    mAP = calculate_ap(precision, recall)  # Compute mean average precision
    
    accuracy = all_tp[-1] / float(total_fn + total_pres) if (total_fn + total_pres) > 0 else 0
    far = all_fp[-1] / float(all_fp[-1] + all_tp[-1] + total_fn) if (all_fp[-1] + all_tp[-1] + total_fn) > 0 else 0
    
    return precision[-1], recall[-1], mAP, accuracy, far

# Main evaluation function for detection tasks
def evaluate_detection2(data, pre_box=False, is_box=False, is_obb=False):
    re_dict = {}
    ious = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # List of IoU thresholds
    for iou in ious:
        precision, recall, mAP, accuracy, far = calculate_eval_matrix(data, iou, pre_box, is_box, is_obb)

        # Store metrics for each IoU threshold
        re_dict[f'FAR@{iou}'] = far * 100.0
        re_dict[f'Recall@{iou}'] = recall * 100.0 if not isinstance(recall, np.ndarray) else 0.0
        re_dict[f'Precision@{iou}'] = precision * 100.0 if not isinstance(precision, np.ndarray) else 0.0
        re_dict[f'mAP@{iou}'] = mAP * 100.0

    return re_dict
