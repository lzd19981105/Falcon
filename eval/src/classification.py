import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.metrics import specificity_score  # Specificity metric, but unused in the current code.
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, 
    classification_report, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import MultiLabelBinarizer  # For handling multi-label classification tasks.

# Custom function to calculate accuracy for multi-label tasks using Intersection over Union (IoU).
def multi_label_accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        # Calculate IoU for each sample and accumulate.
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]  # Return the average accuracy across all samples.

# Function for evaluating single-label classification metrics.
def single_label(y_true_bin, y_pred_bin, class_names):
    # Calculate average accuracy, precision, recall, and F1-score (macro average).
    accuracy_avg = accuracy_score(y_true_bin, y_pred_bin)
    precision_macro = precision_score(y_true_bin, y_pred_bin, average='macro')
    recall_macro = recall_score(y_true_bin, y_pred_bin, average='macro')
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro')  

    # Calculate per-class precision, recall, and F1-score (no averaging).
    precision_mi = precision_score(y_true_bin, y_pred_bin, average=None)
    recall_mi = recall_score(y_true_bin, y_pred_bin, average=None)
    f1_mi = f1_score(y_true_bin, y_pred_bin, average=None)

    # Store per-class metrics in a dictionary.
    micro_metrics = {}
    for class_name, p, r, f in zip(class_names, precision_mi, recall_mi, f1_mi):
        micro_metrics[class_name] = {'Precision': p, 'Recall': r, 'F1-score': f}

    # Return overall metrics and per-class metrics.
    return accuracy_avg, precision_macro, recall_macro, f1_macro, micro_metrics

# Function for evaluating multi-label classification metrics.
def multi_label(y_true_bin, y_pred_bin, class_names):
    # Custom multi-label accuracy using IoU.
    accuracy = multi_label_accuracy(y_true_bin, y_pred_bin)
    # Calculate average precision, recall, and F1-score across samples.
    precision = precision_score(y_true_bin, y_pred_bin, average='samples')
    recall = recall_score(y_true_bin, y_pred_bin, average='samples')
    f1 = f1_score(y_true_bin, y_pred_bin, average='samples')

    # Calculate per-class precision, recall, and F1-score.
    precision_mi = precision_score(y_true_bin, y_pred_bin, average=None)
    recall_mi = recall_score(y_true_bin, y_pred_bin, average=None)
    f1_mi = f1_score(y_true_bin, y_pred_bin, average=None)

    # Store metrics for each class in a dictionary.
    micro_metrics = {}
    for class_name, p, r, f in zip(class_names, precision_mi, recall_mi, f1_mi):
        if ',' not in class_name:  # Exclude compound classes (if any).
            micro_metrics[class_name] = {'Precision': p, 'Recall': r, 'F1-score': f}

    # Return overall metrics and per-class metrics.
    return accuracy, precision, recall, f1, micro_metrics

# Main function for evaluating classification tasks (both single-label and multi-label).
def evaluate_classification(y_true, y_pred, class_names=''):
    # Determine if the task is multi-label by checking for commas in the labels.
    is_multi_label = any(',' in yt for yt in y_true)

    # Convert each string of classes into a set of unique classes.
    y_true = [set(yt.strip().split(', ')) for yt in y_true]
    y_pred = [set(yp.strip().split(', ')) for yp in y_pred]

    # Identify classes in y_pred that are not in class_names.
    unique_pred_classes = set().union(*y_pred)
    extra_classes = unique_pred_classes - set(class_names)

    # Handle unexpected classes by renaming them to 'dummy'.
    if extra_classes:
        print(f"Found classes in y_pred not in class_names: {extra_classes}. Renaming to 'dummy'.")
        y_pred = [{('dummy' if cls in extra_classes else cls) for cls in yp} for yp in y_pred]
        class_names = class_names + ['dummy']  # Add 'dummy' to the class names.

    # Convert true and predicted labels into one-hot encoded format using MultiLabelBinarizer.
    mlb = MultiLabelBinarizer(classes=class_names)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    # Call appropriate evaluation function based on task type.
    if is_multi_label:
        print("Multi-label task detected.")
        return multi_label(y_true_bin, y_pred_bin, class_names)
    else:
        print("Single-label task detected.")
        return single_label(y_true_bin, y_pred_bin, class_names)
