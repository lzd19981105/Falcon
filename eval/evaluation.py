import numpy as np
import json
import argparse
import os
import re
import shutil
from datetime import datetime
import warnings

# Suppress warnings to clean up output
warnings.filterwarnings("ignore")

# Import evaluation functions for different tasks
from src.captioning import evaluate_captioning
from src.classification import evaluate_classification
from src.segmentation import evaluate_segmentation
from src.detection import evaluate_detection
from src.detection_obb import evaluate_detection2
from src.vqa import evaluate_vqa
import const

# Load constants from the `const` module
temp_folder = const.TEMP_FOLDER
cls_tags = const.CLS_TAGS
cap_tags = const.CAP_TAGS
vqa_tags = const.VQA_TAGS
det_tags = const.DET_TAGS
seg_tags = const.SEG_TAGS

# Function to extract regions of interest (ROI) using a regex pattern
def extract_roi(input_string, pattern=r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\|<(\d+)>"):
    # Use regular expression to extract ROI groups
    matches = re.findall(pattern, input_string)
    return [match for match in matches]

# Function to round a number up to the nearest power of 10
def round_up_to_nearest(x):
    if x <= 0:
        return 0  # Return 0 for non-positive numbers
    magnitude = 10 ** (len(str(x)) - 1)
    return x if x == magnitude else magnitude * 10

# Prepare detection data format for evaluation
def prepare_detection_format(file_name='mPLUG-Owl2/json/en_mPLUG-Owl2_rs_reg_det.json', analytic=False):
    # Create a unique folder name based on the current timestamp
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    print("save HBB", folder_name)
    temp_dir = temp_folder + folder_name + '/det/'
    
    # Load the JSON file containing data
    with open(file_name) as file_json:
        json_obj = json.load(file_json)
    
    # Extract task information
    try:
        info_json = json_obj['info']
        result_task = info_json['task']
    except:
        result_task = json_obj['task']
    data_json = json_obj['data']

    # Clear old data directories if they exist
    for subfolder in ["ground-truth", "detection-results"]:
        path = temp_dir + subfolder
        if os.path.exists(path):
            shutil.rmtree(path)

    # Create new directories for this test
    for subfolder in ["ground-truth", "detection-results"]:
        os.makedirs(temp_dir + subfolder, exist_ok=True)

    # Variables to determine scaling factors for ground truth and predictions
    max_answer = max_gt = 0
    multiplyer_ans = multiplyer_gt = 1

    # Determine the maximum values of ground truth and predictions
    for data_j in data_json:
        answer_roi = data_j['answer']
        gt_roi = data_j['gt']
        if not isinstance(gt_roi, list):  
            gt_roi = extract_roi(gt_roi.replace(" ", ""), pattern=const.ROI_PATTERN)
        if not isinstance(answer_roi, list):
            answer_roi = extract_roi(answer_roi.replace(" ", ""), pattern=const.ROI_PATTERN)
        
        max_gt = max(max_gt, *(int(max(gt)) for gt in gt_roi))
        max_answer = max(max_answer, *(int(max(ans)) for ans in answer_roi))

    # Adjust scaling factors based on the maximum values
    max_answer = round_up_to_nearest(max_answer)
    max_gt = round_up_to_nearest(max_gt)
    if max_gt > max_answer:
        multiplyer_ans = max_gt / max_answer
    elif max_gt < max_answer:
        multiplyer_gt = max_answer / max_gt

    # Optional: Force scaling factors to 1 for testing purposes
    if True:
        multiplyer_ans = multiplyer_gt = 1

    # Process each data entry
    for data_j in data_json:
        class_name = 'PIPE'
        image_file = data_j['image']
        if isinstance(image_file, list):
            image_file = image_file[0]
        
        image_name = os.path.splitext(os.path.basename(image_file))[0]

        # Open files for ground truth and detection results
        ground_file = open(temp_dir + "ground-truth/" + image_name + ".txt", 'a')
        res_file = open(temp_dir + "detection-results/" + image_name + '.txt', 'a')
        
        answer_roi = data_j['answer']
        gt_roi = data_j['gt']
        if not isinstance(gt_roi, list):
            gt_roi = extract_roi(gt_roi.replace(" ", ""), pattern=const.ROI_PATTERN)
        if not isinstance(answer_roi, list):
            answer_roi = extract_roi(answer_roi.replace(" ", ""), pattern=const.ROI_PATTERN)

        # Optional: Modify class name based on analytics
        if analytic:
            class_name += 's' if len(gt_roi) > 2 else f'{len(gt_roi)}_{class_name}'

        # Write detection results to respective files
        HBB = True
        for a_roi in answer_roi:
            cx = (int(a_roi[0]) + int(a_roi[2])) / 2
            cy = (int(a_roi[1]) + int(a_roi[3])) / 2
            wi = int(a_roi[2]) - int(a_roi[0])
            hi = int(a_roi[3]) - int(a_roi[1])

            write_result = f"{class_name} 0.9 {int(a_roi[0]) * multiplyer_ans} {int(a_roi[1]) * multiplyer_ans} {int(a_roi[2]) * multiplyer_ans} {int(a_roi[3]) * multiplyer_ans}"
            res_file.write(write_result + "\n")
        
        for a_roi in gt_roi:
            write_result = f"{class_name} {int(a_roi[0]) * multiplyer_gt} {int(a_roi[1]) * multiplyer_gt} {int(a_roi[2]) * multiplyer_gt} {int(a_roi[3]) * multiplyer_gt}"
            ground_file.write(write_result + "\n")

        ground_file.close()
        res_file.close()

    return temp_dir + "ground-truth/", temp_dir + "detection-results/", temp_dir, temp_folder + folder_name



def read_json_result(file_name):
    # Open and read the JSON file
    file_json = open(file_name)
    json_obj = json.load(file_json)

    # Extract basic information: task, model, and dataset
    try:
        info_json = json_obj['info']
        result_task = info_json['task']
        result_model = info_json['model']
        result_dataset = info_json['dataset']
    except:
        # Fallback if the structure does not include 'info'
        result_task = json_obj['task']
        result_model = json_obj['model']
        result_dataset = json_obj['dataset']

    # Initialize variables for processing the data
    data_json = json_obj['data']
    y_true = []  # Ground truth labels
    y_pred = []  # Predicted labels
    class_names = []  # Unique class names
    isCAP = False  # Flag for captioning tasks
    isList = False  # Flag for list-based tasks (e.g., segmentation)
    image_size = []  # Store image sizes for segmentation tasks

    # Determine task type and set appropriate flags
    if result_task in cap_tags:  # Captioning tasks
        isCAP = True
    elif result_task in seg_tags:  # Segmentation tasks
        isList = True

    # Process each data item in the JSON
    for data_i in data_json:
        # Extract and normalize ground truth labels
        try:
            if isCAP:
                y_true.append(data_i['gt'].lower())  # Lowercase for consistency
            elif isList:
                y_true.append(data_i['gt'])  # Keep as list for segmentation
            else:
                y_true.append(data_i['gt'].lower().replace(".", ""))  # Remove special characters
        except:
            # Handle missing ground truth labels
            if isCAP:
                y_true.append(str(data_i['gt']).lower())
            elif isList:
                y_true.append(data_i['gt'])
            else:
                y_true.append(str(data_i['gt']).lower().replace(".", "").replace("<", ""))

        # Extract and normalize predictions
        try:
            if isCAP:
                y_pred.append(data_i['answer'].lower())
            elif isList:
                y_pred.append(data_i['answer'])  # List for segmentation tasks
                image_size.append(data_i['crop'])  # Add image size for evaluation
            else:
                y_pred.append(data_i['answer'].lower().replace(".", "").replace("<", ""))
        except:
            # Handle missing predictions
            if isCAP:
                y_pred.append(str(data_i['answer']).lower())
            elif isList:
                y_pred.append(data_i['answer'])
                image_size.append(data_i['crop'])
            else:
                y_pred.append(str(data_i['answer']).lower().replace(".", "").replace("<", ""))

        # Populate unique class names for classification tasks
        try:
            if data_i['gt'].lower().replace(".", "") not in class_names:
                class_names.append(data_i['gt'].lower().replace(".", ""))
        except:
            continue

    # Perform evaluation based on the task type
    if result_task in cls_tags:
        # Evaluate classification tasks
        accuracy, precision_macro, recall_macro, f1_macro, micro = evaluate_classification(y_true, y_pred, class_names)
        result_dic = {
            'info': {
                'task': result_task,
                'model': result_model,
                'dataset': result_dataset
            },
            'results': {
                'accuracy': accuracy,
                'precision': precision_macro,
                'recall': recall_macro,
                'f1': f1_macro,
                'micro': micro
            }
        }
    elif result_task in cap_tags:
        # Evaluate captioning tasks
        bleu, metero_score, rouge_score, rouge_l, CIDEr = evaluate_captioning(y_true, y_pred)
        result_dic = {
            'info': {
                'task': result_task,
                'model': result_model,
                'dataset': result_dataset
            },
            'results': {
                'bleu': bleu,
                'metero': metero_score,
                'rouge': rouge_score,
                'rouge_l': rouge_l,
                'CIDEr': CIDEr
            }
        }
    elif result_task in seg_tags:
        # Evaluate segmentation tasks
        m_dice, m_iou, m_vc8, m_vc16 = evaluate_segmentation(y_true, y_pred, image_size)
        result_dic = {
            'info': {
                'task': result_task,
                'model': result_model,
                'dataset': result_dataset
            },
            'results': {
                'mdice': m_dice,
                'miou': m_iou
                # Optional metrics (commented out for now)
                # 'mvc8': m_vc8,
                # 'mvc16': m_vc16
            }
        }
    elif result_task in det_tags:
        # Evaluate detection tasks
        isOBB = False  # Flag for oriented bounding boxes
        isBox = False  # Flag for horizontal bounding boxes
        pre_box = False  # Flag for prediction format (list or not)

        # Check if task involves oriented bounding boxes (OBB)
        if "obb" in result_task.lower():
            isOBB = True

        # Determine bounding box format (list or direct bounding box)
        answer_roi = data_json[0]['answer']
        gt_roi = data_json[0]['gt']
        if not isinstance(gt_roi, list):
            isBox = True
        if not isinstance(answer_roi, list):
            pre_box = True

        detection_result = {}
        iou_res = {}

        # Process based on detection task type
        if isOBB:
            detection_result = evaluate_detection2(data_json, pre_box=pre_box, is_box=isBox, is_obb=isOBB)
        elif "vg" in result_task.lower():
            detection_result = evaluate_detection2(data_json, pre_box=pre_box, is_box=isBox, is_obb=isOBB)
        else:
            # Fallback for horizontal bounding boxes (HBB)
            gt_roi_folder, res_roi_folder, temp_save_folder, master_folder = prepare_detection_format(file_name)
            ious = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for iou in ious:
                det_map = evaluate_detection(
                    GT_PATH=gt_roi_folder,
                    DR_PATH=res_roi_folder,
                    TEMP_FILES_PATH=temp_save_folder + "temps",
                    output_files_path=temp_save_folder + "output_detection/",
                    iou=iou,
                    draw_plot=False
                )
                iou_res['mAP@' + str(iou)] = det_map

            # Clean up temporary files and directories
            for folder in [gt_roi_folder, res_roi_folder, temp_save_folder, master_folder]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)

        result_dic = {
            'info': {
                'task': result_task,
                'model': result_model,
                'dataset': result_dataset
            },
            'results': {**detection_result, **iou_res}
        }
    elif result_task in vqa_tags:
        # Evaluate visual question answering (VQA) tasks
        accuracy, precision_macro, recall_macro, f1_macro, micro = evaluate_classification(y_true, y_pred, class_names)
        accuracy, precision, recall, f1 = evaluate_vqa(y_true, y_pred)
        result_dic = {
            'info': {
                'task': result_task,
                'model': result_model,
                'dataset': result_dataset
            },
            'results': {
                'accuracy': accuracy,
                'f1': f1
            }
        }
    else:
        # Handle unknown or unsupported tasks
        print('Error:', result_task)
        result_dic = {
            'info': {
                'task': result_task,
                'model': result_model,
                'dataset': result_dataset
            }
        }

    # Return the evaluation results
    return result_dic

def evaluation(args):
    # Extract the folder where the evaluation files are stored
    evaluation_folder = args.evaluation_folder
    debuging = False  # Flag for debugging mode
    isReplace = args.isReplace  # Flag to determine if existing files should be replaced

    # Enable debugging if specified in arguments
    if args.debug == "yes":
        debuging = True

    # Define the directory where evaluation results will be saved
    save_evaluation_result = args.save_path
    save_result_folder_dir = save_evaluation_result + args.model_name + "/"

    # Create the result directory if it doesn't already exist
    if not os.path.exists(save_result_folder_dir):
        os.makedirs(save_result_folder_dir)

    # Get a list of all files already saved in the result directory
    results_files = os.listdir(save_result_folder_dir)

    # Check if an evaluation folder is specified
    if evaluation_folder == '':
        # If no folder is specified, use a single file path for evaluation
        evaluation_path = args.evaluation_file

        if evaluation_path == '':
            # If no file is specified, prompt the user to provide input
            result_dic = 'please specify --evaluation-file or --evaluation-folder'
        else:
            if not debuging:
                # Get the file name from the evaluation path
                file_name = evaluation_path.split('/')[-1]
                save_file_name = '[' + args.model_name + "]" + file_name

                # Skip evaluation if the result file already exists and replacement is not allowed
                if save_file_name in os.listdir(save_result_folder_dir) and not isReplace:
                    print(f'{file_name} finish already!')
                else:
                    # Read and process the JSON result
                    result_dic = read_json_result(evaluation_path)
                    print(f'{result_dic}')

                    # Save the processed result as a JSON file
                    final_json = json.dumps(result_dic)
                    with open(save_result_folder_dir + '[' + args.model_name + "]" + file_name, "w") as outfile:
                        outfile.write(final_json)
            else:
                # Debugging mode: Process and display results without saving
                result_dic = read_json_result(evaluation_path)
                file_name = evaluation_path.split('/')[-1]
                print(f'{result_dic}')
                final_json = json.dumps(result_dic)
    else:
        # If an evaluation folder is specified, process all JSON files in the folder
        with open(save_result_folder_dir + "log.txt", "w") as log_file:
            log_file.write("missing file:\n")

        # Iterate through each file in the evaluation folder
        for jsons in os.listdir(evaluation_folder):
            if jsons.endswith(".json"):  # Process only JSON files
                save_file_name = '[' + args.model_name + "]" + jsons

                # Skip evaluation if the result file already exists and replacement is not allowed
                if save_file_name in os.listdir(save_result_folder_dir) and not isReplace:
                    print(f'{jsons} finish already!')
                else:
                    # Construct the full path to the evaluation file
                    evaluation_path = evaluation_folder + "/" + jsons
                    print(f'Evaluate: {evaluation_path}')

                    if not debuging:
                        try:
                            # Read and process the JSON result
                            result_dic = read_json_result(evaluation_path)

                            print('done read')
                            print(f'{result_dic}')
                            print()

                            # Save the processed result as a JSON file
                            final_json = json.dumps(result_dic)
                            with open(save_result_folder_dir + '[' + args.model_name + "]" + jsons, "w") as outfile:
                                outfile.write(final_json)
                        except Exception as e:
                            # Log any errors encountered during processing
                            with open(save_result_folder_dir + "log.txt", "a") as log_file:
                                log_file.write(f"Error processing {jsons}: {e}\n")
                    else:
                        # Debugging mode: Process and display results without saving
                        result_dic = read_json_result(evaluation_path)

                        print('done read')
                        print(f'{result_dic}')
                        print()

                        final_json = json.dumps(result_dic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-file", type=str, default="")
    parser.add_argument("--evaluation-folder", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--isReplace", type=bool, default=False)
    parser.add_argument("--save_path", type=str, default='results/20241010/')
    parser.add_argument("--debug", type=str, default='no')
    args = parser.parse_args()

    evaluation(args)
