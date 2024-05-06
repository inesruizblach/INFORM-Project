# IMPORTS
import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
from matplotlib import patches
from ultralytics import YOLO

import edge_sam as edgesam
import mobile_sam as mobilesam

from utilties import load_label_names_from_text_file, load_image_and_annotation, show_mask, class_id_to_color


def visualise_segmentation_masks(original_images, annotation_images, yolo_images, pred_seg_masks_gt,
                                 pred_seg_masks_yolo, gt_bboxes_list, label_names):
    """
    Visualizes a set of images alongside their annotations, predictions and segmentation masks.

    The function plots original images, ground truth bounding boxes, YOLO predictions, and segmentation masks
    from ground truth and YOLO models across multiple subplots.

    Args:
        original_images (List[np.ndarray]): List of original RGB images.
        annotation_images (List[np.ndarray]): List of corresponding annotation masks.
        yolo_images (List[torch.Tensor]): List of images with YOLO model predictions.
        pred_seg_masks_gt (List[dict]): List of dictionaries containing predicted segmentation masks using ground truth boxes.
        pred_seg_masks_yolo (List[dict]): List of dictionaries containing predicted segmentation masks using YOLO detected boxes.
        gt_bboxes_list (List[list]): List containing ground truth bounding boxes for each image.
        label_names (Dict[int, str]): Dictionary mapping label IDs to their respective names.
    """
    for i, (original_image, annotation_image, yolo, masks_gt, masks_yolo, bboxes) in enumerate(
            zip(original_images, annotation_images, yolo_images, pred_seg_masks_gt, pred_seg_masks_yolo,
                gt_bboxes_list)):
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(30, 5))

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Original image with GT bounding boxes
        axes[1].imshow(original_image)
        class_ids = np.unique(annotation_image)
        for class_id in class_ids:
            if class_id == 0:  # Skip background
                continue
            mask = annotation_image == class_id
            box = np.argwhere(mask)
            if box.size > 0:
                y_min, x_min = box.min(axis=0)
                y_max, x_max = box.max(axis=0)
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2,
                                         edgecolor=class_id_to_color(class_id), facecolor='none')
                axes[1].add_patch(rect)
                label_name = label_names.get(class_id, 'Unknown')
                # Adjust text position if outside plot boundaries
                text_x = x_min
                text_y = y_min - 10
                if text_x < 0:
                    text_x = 0
                if text_y < 0:
                    text_y = 0
                axes[1].text(text_x, text_y, label_name, color='white', fontsize=12)
        axes[1].set_title('GT Bounding Boxes')
        axes[1].axis('off')

        # YOLO image
        axes[2].imshow(yolo)
        axes[2].set_title('YOLO Predictions')
        axes[2].axis('off')

        # Background with predicted YOLO masks
        background = np.ones_like(original_image) * 255
        for class_id, mask in masks_yolo.items():
            show_mask(mask, class_id, axes[3])
        axes[3].set_title('Predicted Masks - YOLO Detections')
        axes[3].axis('off')

        # Background with predicted GT masks
        axes[4].imshow(background)
        for class_id, mask in masks_gt.items():
            show_mask(mask, class_id, axes[4])
        axes[4].set_title('Predicted Masks - GT Boxes')
        axes[4].axis('off')

        # Retrieve annotation image masks
        axes[5].imshow(background)
        for class_id in np.unique(annotation_image):
            if class_id == 0:  # Skip background
                continue
            gt_mask = (annotation_image == class_id).astype(np.uint8)
            show_mask(gt_mask, class_id, axes[5])
        axes[5].set_title('Ground Truth Masks')
        axes[5].axis('off')

        plt.tight_layout()
        plt.show()


def extract_bounding_boxes_from_mask(annotation_image):
    """
    Extracts bounding boxes from the annotation mask for each class ID present in the mask.

    Args:
        annotation_image (np.ndarray): Annotated mask image where each pixel value corresponds to a class ID.

    Returns:
        class_bboxes (dict): A dictionary where keys are class IDs and values are lists of bounding boxes specified as [x, y, width, height].
        gt_detections (list): List of dictionaries containing bounding box coordinates and class IDs.
    """
    unique_classes = np.unique(annotation_image)
    class_bboxes = defaultdict(list)
    for class_id in unique_classes:
        if class_id == 0:  # Typically 0 is the background class and can be ignored
            continue
        mask = np.where(annotation_image == class_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            class_bboxes[class_id].append([x, y, x + w, y + h])

    gt_detections = []
    for class_id, bboxes in class_bboxes.items():
        for bbox in bboxes:
            x, y, x_plus_w, y_plus_h = bbox
            gt_detections.append({'box_coords': bbox, 'class_id': class_id})

    return class_bboxes, gt_detections


def detect_objects_yolo(yolo_model, image_path, yolo_total_inference_time):
    """
    Detects objects in an image using a YOLO model, recording inference times.

    Args:
        yolo_model (YOLO): The YOLO model to use for detection.
        image_path (str): The path to the image file.
        yolo_total_inference_time (float): Accumulated total inference time up to this call.

    Returns:
        detected_objects (list): A list of dictionaries, each containing details of detected objects including
                                 bounding box coordinates, class ID, and confidence score.
        yolo_image (PIL.Image): An image with detection results plotted.
        yolo_total_inference_time (float): Updated total inference time including this call.
    """
    results = yolo_model(source=image_path, show_labels=True)

    detected_objects = []
    yolo_image = None
    for r in results:
        im_array = r.plot(line_width=1, font_size=9)  # plot a BGR numpy array of predictions
        yolo_image = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

        for box in r.boxes:
            label = r.names[box.cls[0].item()]
            class_id = int(box.cls)
            box_coords = box.xyxy[0].tolist()
            box_coords = [round(x) for x in box_coords]
            confidence = round(box.conf[0].item(), 2)
            detected_objects.append({
                'box_coords': box_coords,
                'class_id': class_id,
                'confidence': confidence
            })

        # Retrieve and sum the inference time (ms)
        yolo_total_inference_time += r.speed['inference']  # Add the inference time of the current image

    return detected_objects, yolo_image, yolo_total_inference_time


def segment_with_sam(sam_model, sam_predictor, image, detections):
    """
    Segments an image using either EdgeSAM or MobileSAM models, based on provided detections.

    Args:
        sam_model (str): Specifies which SAM model to use ('edgesam' or 'mobilesam').
        sam_predictor (object): SAM predictor instance.
        image (np.ndarray): Image to segment.
        detections (list): List of detections, where each detection is a dictionary with keys 'box_coords' and 'class_id'.

    Returns:
        predicted_masks (dict): A dictionary of predicted masks, indexed by class ID, where each mask is a binary numpy array.
    """
    predicted_masks = {}
    for detection in detections:
        bbox = detection['box_coords']  # Assuming bbox format is [x1, y1, x2, y2]
        class_id = detection['class_id']

        # Convert bounding box to tensor and apply transformations
        input_box = torch.tensor([bbox], dtype=torch.float32, device=sam_predictor.device)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_box, image.shape[:2])

        if sam_model.lower() == 'edgesam':
            predicted_mask, _, _ = sam_predictor.predict_torch(
                features=None,
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                num_multimask_outputs=1,
            )

        if sam_model.lower() == 'mobilesam':
            predicted_mask, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                mask_input=None,
                multimask_output=False,
            )

        # Squeeze out the batch and channel dimensions to get a 2D mask
        mask = predicted_mask.squeeze().detach().cpu().numpy()
        if mask.ndim > 2:
            continue  # If there's still more than 2 dimensions, something is wrong.

        binary_mask = (mask > 0.5).astype(np.uint8)  # Convert to binary mask

        # Resize mask to the original image size
        resized_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Place the resized mask in a full-size mask image
        full_size_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_size_mask = resized_mask

        # Combine masks for the same class ID using logical OR
        if class_id in predicted_masks:
            predicted_masks[class_id] = np.logical_or(predicted_masks[class_id], full_size_mask).astype(np.uint8)
        else:
            predicted_masks[class_id] = full_size_mask

    return predicted_masks


def evaluate_image_metrics(predicted_masks, annotation_image, label_ids, batch_iou_scores_per_class,
                           batch_acc_scores_per_class, batch_total_tp_sum, batch_total_relevant_pixels):
    """
    Evaluates segmentation metrics such as Intersection over Union (IoU) and accuracy for predicted masks against ground truth masks.

    Args:
        predicted_masks (dict): A dictionary of predicted masks indexed by class ID, where each mask is a binary numpy array.
        annotation_image (np.ndarray): The ground truth annotation mask.
        label_ids (list): A list of valid class IDs to consider in evaluation.
        batch_iou_scores_per_class (dict): Accumulated IoU scores for each class.
        batch_acc_scores_per_class (dict): Accumulated accuracy scores for each class.
        batch_total_tp_sum (int): Total true positives accumulated across the batch.
        batch_total_relevant_pixels (int): Total relevant pixels (non-background) considered in the batch.

    Returns:
        tuple:
            - batch_iou_scores_per_class (dict): Updated IoU scores post evaluation.
            - batch_acc_scores_per_class (dict): Updated accuracy scores post evaluation.
            - batch_total_tp_sum (int): Updated total true positives.
            - batch_total_relevant_pixels (int): Updated total relevant pixels.
    """
    # Evaluate masks and accumulate metrics within the batch
    for class_id, pred_mask in predicted_masks.items():
        if class_id in label_ids:
            gt_mask = (annotation_image == class_id).astype(np.uint8)

            pred_mask_flat = pred_mask.flatten()
            gt_mask_flat = gt_mask.flatten()

            # Calculate relevant masks and metrics
            intersection = np.logical_and(pred_mask_flat, gt_mask_flat)
            union = np.logical_or(pred_mask_flat, gt_mask_flat)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
            batch_iou_scores_per_class[class_id].append(iou)

            tp = np.sum((gt_mask_flat == 1) & (pred_mask_flat == 1))
            fn = np.sum((gt_mask_flat == 1) & (pred_mask_flat == 0))
            acc = tp / (tp + fn) if (tp + fn) > 0 else 0
            batch_acc_scores_per_class[class_id].append(acc)

            batch_total_tp_sum += tp
            batch_total_relevant_pixels += np.sum(gt_mask_flat)

    return batch_iou_scores_per_class, batch_acc_scores_per_class, batch_total_tp_sum, batch_total_relevant_pixels


def compute_batch_metrics(aggregated_metrics, batch_iou_scores_per_class, batch_acc_scores_per_class,
                          batch_total_tp_sum,
                          batch_total_relevant_pixels, label_ids):
    """
    Computes mean metrics for a batch of images including mean Intersection over Union (IoU), mean accuracy, and accuracy over all pixels.

    Args:
        aggregated_metrics (dict): Dictionary storing lists of metrics aggregated across all processed batches.
        batch_iou_scores_per_class (dict): IoU scores for the current batch indexed by class ID.
        batch_acc_scores_per_class (dict): Accuracy scores for the current batch indexed by class ID.
        batch_total_tp_sum (int): Total true positives from the current batch.
        batch_total_relevant_pixels (int): Total relevant pixels considered in the current batch.
        label_ids (list): List of class IDs considered for metric computation.

    Returns:
        aggregated_metrics (dict): Updated aggregated metrics after including the current batch's results.
    """
    # Compute metrics for the current batch
    final_iou_scores = [np.mean(batch_iou_scores_per_class[class_id]) for class_id in label_ids if
                        batch_iou_scores_per_class[class_id]]
    final_acc_scores = [np.mean(batch_acc_scores_per_class[class_id]) for class_id in label_ids if
                        batch_acc_scores_per_class[class_id]]
    batch_mIoU = np.mean(final_iou_scores) if final_iou_scores else 0
    batch_mAcc = np.mean(final_acc_scores) if final_acc_scores else 0
    batch_aAcc = batch_total_tp_sum / batch_total_relevant_pixels if batch_total_relevant_pixels else 0

    # Store batch metrics
    aggregated_metrics['Mean IoU (mIoU)'].append(batch_mIoU)
    aggregated_metrics['Mean Accuracy (mAcc)'].append(batch_mAcc)
    aggregated_metrics['Accuracy over all pixels (aAcc)'].append(batch_aAcc)

    print(f"Batch Metrics: IoU: {batch_mIoU}, Accuracy: {batch_mAcc}, aAcc: {batch_aAcc}")

    return aggregated_metrics


def chunk_list(input_list, chunk_size):
    """
    Yields successive chunks from a list.

    Args:
        input_list (list): The list to be divided into chunks.
        chunk_size (int): The size of each chunk.

    Yields:
        list: A chunk of the input list of size `chunk_size` or smaller if the list ends.
    """
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]


def yolo_edge_sam_predictor(image_filenames, image_dir, ann_dir, categories_txt_file_path, sam_model, yolo_model_path,
                            batch_size=10, plot_results=False):
    """
    Orchestrates the entire process of image segmentation using YOLO for object detection and SAM for image segmentation on specified image sets.

    Processes images in batches, applying YOLO detection and SAM segmentation, and optionally plots the results.

    Args:
        image_filenames (List[str]): Filenames of the test images.
        image_dir (str): Directory containing the test images.
        ann_dir (str): Directory containing the annotation masks.
        categories_txt_file_path (str): Path to the TSV file containing label names and IDs.
        sam_model (str): Specifies the SAM model to use ('edgesam' or 'mobilesam').
        yolo_model_path (str): Path to the YOLO model weights.
        batch_size (int): Number of images to process in one batch.
        plot_results (bool): Whether to display the results visually after processing.

    Note:
        The function is designed to demonstrate the integration of object detection and segmentation models in a realistic application scenario.
    """
    yolo_model = YOLO(yolo_model_path)

    # Select the computation device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load the specific SAM model based on the parameter
    if sam_model.lower() == 'edgesam':
        # Load the pre-trained EdgeSAM model from checkpoint
        sam_checkpoint = "/content/drive/MyDrive/Dissertation/Weights/edge_sam.pth"
        model_type = "edge_sam"
        edge_sam = edgesam.sam_model_registry[model_type](checkpoint=sam_checkpoint)
        edge_sam = edge_sam.to(device=device)
        sam_predictor = edgesam.SamPredictor(edge_sam)

        # Setting up resize transformation and loading label names
        resize_transform = edgesam.utils.transforms.ResizeLongestSide(edge_sam.image_encoder.img_size)

    if sam_model.lower() == 'mobilesam':
        # Load the pre-trained MobileSAM model
        sam_checkpoint = "/content/drive/MyDrive/Dissertation/Weights/mobile_sam.pt"
        model_type = "vit_t"
        mobile_sam = mobilesam.sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam = mobile_sam.to(device=device)
        mobile_sam.eval()
        sam_predictor = mobilesam.SamPredictor(mobile_sam)

    # Load label names from a text file
    label_names, label_ids = load_label_names_from_text_file(categories_txt_file_path)
    print(f"Loaded labels: {label_names}")
    if next(iter(label_names)) == 1:
        num_classes = len(label_names)
    else:
        num_classes = len(label_names) + 1
    print(num_classes)

    # Initialise inference time counters
    yolo_total_inference_time = 0
    yolo_inference_time = 0
    sam_gt_boxes_total_inference_time = 0
    sam_yolo_boxes_total_inference_time = 0
    load_time_total = 0

    # Initialize dictionary to store aggregated metrics across batches
    aggregated_metrics = {'Mean IoU (mIoU)': [], 'Mean Accuracy (mAcc)': [], 'Accuracy over all pixels (aAcc)': []}
    aggregated_gt_metrics = {'Mean IoU (mIoU)': [], 'Mean Accuracy (mAcc)': [], 'Accuracy over all pixels (aAcc)': []}

    # Iterate over the filenames in chunks of size `batch_size`
    for chunk_filenames in chunk_list(image_filenames, batch_size):

        # Initialize metrics for the batch using GT boxes
        batch_gt_iou_scores_per_class = {class_id: [] for class_id in label_ids}
        batch_gt_acc_scores_per_class = {class_id: [] for class_id in label_ids}
        batch_gt_total_tp_sum = 0
        batch_gt_total_relevant_pixels = 0

        # Initialize metrics for the batch using YOLO boxes
        batch_iou_scores_per_class = {class_id: [] for class_id in label_ids}
        batch_acc_scores_per_class = {class_id: [] for class_id in label_ids}
        batch_total_tp_sum = 0
        batch_total_relevant_pixels = 0

        # Example of how to prepare data for visualization after processing a batch
        original_images, annotation_images, yolo_images = [], [], []
        pred_seg_masks_gt, pred_seg_masks_yolo, bounding_boxes_with_labels_list = [], [], []

        for image_filename in chunk_filenames:
            # Start inference time counter
            # Measure the time for loading an image and its annotation
            start_load_time = time.time()

            # Load an image and its annotation
            original_image, annotation_image, original_image_path = load_image_and_annotation(image_filename,
                                                                                                       image_dir,
                                                                                                       ann_dir)
            load_time = time.time() - start_load_time
            load_time_total += load_time

            # Extract GT bounding boxes
            gt_bboxes, gt_detections = extract_bounding_boxes_from_mask(annotation_image)

            # Set image for SAM predictor
            sam_predictor.set_image(original_image)

            # Segment with SAM using GT bounding boxes
            start_gt_segmentation_time = time.time()
            predicted_masks_with_gt = segment_with_sam(sam_model, sam_predictor, original_image, gt_detections)
            sam_gt_segmentation_time = time.time() - start_gt_segmentation_time
            sam_gt_boxes_total_inference_time += sam_gt_segmentation_time

            # Evaluate masks with GT boxes and accumulate metrics within the batch
            batch_gt_iou_scores_per_class, batch_gt_acc_scores_per_class, batch_gt_total_tp_sum, batch_gt_total_relevant_pixels = evaluate_image_metrics(
                predicted_masks_with_gt, annotation_image, label_ids,
                batch_gt_iou_scores_per_class, batch_gt_acc_scores_per_class,
                batch_gt_total_tp_sum, batch_gt_total_relevant_pixels)

            # Detect objects with YOLO and measure inference time
            start_detection_time = time.time()
            yolo_detections, yolo_image, yolo_total_inference_time = detect_objects_yolo(yolo_model,
                                                                                         original_image_path,
                                                                                         yolo_total_inference_time)
            detection_time = time.time() - start_detection_time
            yolo_inference_time += detection_time

            # Segment with SAM using YOLO detections
            start_segmentation_time = time.time()
            predicted_masks = segment_with_sam(sam_model, sam_predictor, original_image, yolo_detections)
            segmentation_time = time.time() - start_segmentation_time
            sam_yolo_boxes_total_inference_time += segmentation_time

            # Evaluate masks with YOLO boxes and accumulate metrics within the batch
            batch_iou_scores_per_class, batch_acc_scores_per_class, batch_total_tp_sum, batch_total_relevant_pixels = evaluate_image_metrics(
                predicted_masks, annotation_image, label_ids,
                batch_iou_scores_per_class, batch_acc_scores_per_class,
                batch_total_tp_sum, batch_total_relevant_pixels)

            # Prepare data for visualization after processing a batch
            original_images.append(original_image)
            annotation_images.append(annotation_image)
            yolo_images.append(yolo_image)
            pred_seg_masks_gt.append(predicted_masks_with_gt)
            pred_seg_masks_yolo.append(predicted_masks)
            bounding_boxes_with_labels_list.append(
                [(bbox[0], bbox[1], bbox[2], bbox[3], class_id) for class_id, bboxes in gt_bboxes.items() for bbox in
                 bboxes])

        # Compute metrics for the current batch
        aggregated_gt_metrics = compute_batch_metrics(aggregated_gt_metrics, batch_gt_iou_scores_per_class,
                                                      batch_gt_acc_scores_per_class, batch_gt_total_tp_sum,
                                                      batch_gt_total_relevant_pixels, label_ids)

        # Compute metrics for the current batch
        aggregated_metrics = compute_batch_metrics(aggregated_metrics, batch_iou_scores_per_class,
                                                   batch_acc_scores_per_class, batch_total_tp_sum,
                                                   batch_total_relevant_pixels, label_ids)

        # Visualise predictions against annotation image
        if plot_results:
            visualise_segmentation_masks(original_images, annotation_images, yolo_images, pred_seg_masks_gt,
                                         pred_seg_masks_yolo, bounding_boxes_with_labels_list, label_names)

    # Calculate and print final metrics after processing all batches
    final_gt_mIoU = np.mean(aggregated_gt_metrics['Mean IoU (mIoU)'])
    final_gt_mAcc = np.mean(aggregated_gt_metrics['Mean Accuracy (mAcc)'])
    final_gt_aAcc = np.mean(aggregated_gt_metrics['Accuracy over all pixels (aAcc)'])

    # Calculate and print final metrics after processing all batches
    final_mIoU = np.mean(aggregated_metrics['Mean IoU (mIoU)'])
    final_mAcc = np.mean(aggregated_metrics['Mean Accuracy (mAcc)'])
    final_aAcc = np.mean(aggregated_metrics['Accuracy over all pixels (aAcc)'])

    # Print total times
    print(f"Total loading input and annotation images time: {load_time_total:.2f} seconds")

    print("\nSegmentation Pipeline using YOLO detections:")
    yolo_total_inference_time_seconds = yolo_total_inference_time / 1000
    print(f"Total YOLO detection time: {yolo_total_inference_time_seconds:.2f} seconds")
    print(f"Total SAM segmentation time using YOLO boxes: {sam_yolo_boxes_total_inference_time:.2f} seconds")
    total_time = load_time_total + yolo_total_inference_time + sam_yolo_boxes_total_inference_time
    minutes, seconds = divmod(total_time, 60)
    print(f"Overall processing time with YOLO: {int(minutes)} minutes and {seconds:.2f} seconds")
    print("\nMETRICS with YOLO:")
    print(f"Mean IoU (mIoU): {final_mIoU}")
    print(f"Mean Accuracy (mAcc): {final_mAcc}")
    print(f"Accuracy over all pixels (aAcc): {final_aAcc}")

    print("\n----------------------------")

    print("\nSegmentation Pipeline using Ground Truth bounding boxes:")
    print(f"Total SAM segmentation time using GT boxes: {sam_gt_boxes_total_inference_time:.2f} seconds")
    total_gt_time = load_time_total + yolo_total_inference_time + sam_gt_boxes_total_inference_time
    minutes, seconds = divmod(total_gt_time, 60)
    print(f"Overall processing time with GT: {int(minutes)} minutes and {seconds:.2f} seconds")
    print("\nMETRICS with GT:")
    print(f"Mean IoU (mIoU): {final_gt_mIoU}")
    print(f"Mean Accuracy (mAcc): {final_gt_mAcc}")
    print(f"Accuracy over all pixels (aAcc): {final_gt_aAcc}")


def main():
    parser = argparse.ArgumentParser(description="Run YOLO+EdgeSAM/MobileSAM Predictor")
    parser.add_argument("--image_filenames", nargs='*', default=None,
                        help="Optional list of specific image filenames to process")
    parser.add_argument("image_dir", type=str, help="Directory containing the test images")
    parser.add_argument("ann_dir", type=str, help="Directory containing the annotation masks")
    parser.add_argument("categories_txt_file_path", type=str,
                        help="Path to the TXT file containing label IDs and names")
    parser.add_argument("sam_model", type=str, help="Specifies the SAM model to use ('edgesam' or 'mobilesam')")
    parser.add_argument("yolo_model_path", type=str, help="Path to the YOLOv8 model weights")
    parser.add_argument("--batch_size", type=int, default=200, help="Number of images to process in one batch")
    parser.add_argument("--plot_results", action='store_true',
                        help="Whether to display the results visually after processing")

    args = parser.parse_args()

    # Check if specific filenames are provided or default to retrieving all filenames from the image directory
    if args.image_filenames:
        image_filenames = args.image_filenames
    else:
        # List all image filenames in the specified directory
        image_filenames = [f.name for f in Path(args.image_dir).glob('*.jpg')]

    yolo_edge_sam_predictor(
        image_filenames=image_filenames,
        image_dir=args.image_dir,
        ann_dir=args.ann_dir,
        categories_txt_file_path=args.categories_txt_file_path,
        sam_model=args.sam_model,
        yolo_model_path=args.yolo_model_path,
        batch_size=args.batch_size,
        plot_results=args.plot_results
    )


if __name__ == "__main__":
    main()
