# IMPORTS
import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import jaccard_score
from skimage.transform import resize
from PIL import Image
from matplotlib import patches
from ultralytics import YOLO
import matplotlib.pyplot as plt

from utilties import load_label_names_from_text_file, load_image_and_annotation, show_mask, class_id_to_color


def plot_image_with_bounding_boxes(original_image, bounding_boxes_with_labels, label_names, ax):
    """
    Plot an image with bounding boxes and corresponding labels.

    Args:
        original_image (np.ndarray): The image on which to plot bounding boxes.
        bounding_boxes_with_labels (list): A list of tuples, each containing a label ID and bounding box coordinates as (label_id, x, y, w, h).
        label_names (dict): Dictionary mapping label IDs to label names.
        ax (matplotlib.axes.Axes): The axes on which to plot the image and bounding boxes.
    """
    ax.imshow(original_image)
    # Plot each bounding box
    for label_id, x, y, w, h in bounding_boxes_with_labels:
        if label_id != 0:
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=class_id_to_color(label_id), facecolor='none')
            ax.add_patch(rect)

            # Retrieve the class label name using the label_id index
            label_name = label_names.get(label_id, "Unknown")
            # Display the class label name near the bounding box
            ax.text(x, y - 10, label_name, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))


def visualize_segmentation_masks(original_images, annotation_images, yolo_images, yolo_masks,
                                 bounding_boxes_with_labels_list, label_names):
    """
    Visualize results for a sets of input images along with their annotations, model predictions, and bounding boxes.

    Args:
        original_images (List[np.ndarray]): List of original images.
        annotation_images (List[np.ndarray]): List of corresponding annotation masks.
        yolo_images (List[torch.Tensor]): List of images with predictions from YOLO model.
        yolo_masks (List[np.ndarray]): List of predicted masks from YOLO.
        bounding_boxes_with_labels_list (List[list]): List of lists containing bounding boxes and labels for each image.
        label_names (Dict[int, str]): Dictionary mapping label IDs to label names.
    """
    image_annotation_info_zip = zip(original_images, annotation_images, yolo_images, yolo_masks,
                                    bounding_boxes_with_labels_list)
    for i, (original_image, annotation_image, yolo_image, yolo_mask, bounding_boxes_with_labels) in enumerate(
            image_annotation_info_zip):
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(14, 6))

        # Plot input image
        axes[0].imshow(original_image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Plot input image with gt bounding boxes and labels
        axes[1].imshow(original_image)
        plot_image_with_bounding_boxes(original_image, bounding_boxes_with_labels, label_names, axes[1])
        axes[1].set_title('GT Box & Labels')
        axes[1].axis('off')

        # Plot the predictions from custom trained YOLOv8 segmentation model
        axes[2].imshow(yolo_image)
        axes[2].set_title('YOLO Image')
        axes[2].axis('off')

        # Plot only the predicted YOLOv8 masks (without boxes and labels)
        axes[3].imshow(yolo_mask)
        axes[3].set_title('YOLO Masks')
        axes[3].axis('off')

        # Plot ground truth masks
        background = np.ones_like(original_image) * 255
        axes[4].imshow(background)
        for label_id in np.unique(annotation_image):
            if label_id == 0:  # Skip background
                continue
            gt_mask = (annotation_image == label_id).astype(np.uint8)
            show_mask(gt_mask, label_id, axes[4])
        axes[4].set_title('Ground Truth Masks')
        axes[4].axis('off')

        plt.tight_layout()
        plt.show()


def extract_bounding_boxes(annotation_image, list_classes):
    """
    Extract bounding boxes from an annotation mask based on specified class IDs.

    Args:
        annotation_image (np.ndarray): The annotation mask.
        list_classes (list): List of class IDs to consider for bounding boxes.

    Returns:
        bounding_boxes (list): A list of tuples, each containing the class label and bounding box coordinates (x, y, width, height).
        label_ids (list): List of class IDs corresponding to the bounding boxes.
    """
    bounding_boxes = []
    label_ids = []
    for label_id in list_classes:
        if label_id != 0:
            binary_mask = (annotation_image == label_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((label_id, x, y, w, h))
                label_ids.append(label_id)
    return bounding_boxes, label_ids


def prepare_yolo_masks_for_metrics(pred_masks, pred_class_ids, orig_shape):
    """
    Process predicted masks for metric calculation by resizing and thresholding.

    Args:
        pred_masks (torch.Tensor): Tensor containing predicted masks.
        pred_class_ids (np.ndarray): Array of class IDs for the predicted masks.
        orig_shape (tuple): Original shape of the images to which masks need to be resized.

    Returns:
        processed_masks_by_class (dict): Dictionary mapping class IDs to processed masks.
    """
    processed_masks_by_class = {}

    if pred_masks is not None:
        mask_array = pred_masks.data.cpu().numpy()  # Ensure tensor is moved to CPU and converted to numpy array

        for mask, class_id in zip(mask_array, pred_class_ids):
            resized_mask = resize(mask, orig_shape, order=0, mode='constant', cval=0, anti_aliasing=False)
            binary_mask = (resized_mask >= 0.5).astype(np.uint8)

            if class_id in processed_masks_by_class:
                # Ensure that dimensions match before applying logical OR
                if processed_masks_by_class[class_id].shape != orig_shape:
                    print(
                        f"Error: Shape mismatch for class {class_id}. Existing shape: {processed_masks_by_class[class_id].shape}, new mask shape: {binary_mask.shape}")
                else:
                    processed_masks_by_class[class_id] |= binary_mask
            else:
                processed_masks_by_class[class_id] = binary_mask
    else:
        return {}

    return processed_masks_by_class


def compute_segmentation_metrics(images, all_pred_masks, label_ids):
    """
    Compute segmentation metrics such as mean IoU and accuracy for each class.

    Args:
        images (list of np.ndarray): List of ground truth annotation masks.
        all_pred_masks (list of dicts): List of dictionaries containing predicted masks for each image.
        label_ids (list): List of class IDs considered for metrics.

    Returns:
        seg_metrics (dict): Dictionary containing calculated metrics (Mean IoU, Mean Accuracy, Accuracy over all pixels).
    """
    iou_scores_per_class = {class_id: [] for class_id in label_ids}
    acc_scores_per_class = {class_id: [] for class_id in label_ids}
    total_tp_sum = 0
    total_relevant_pixels = 0  # Change from total_pixels to total_relevant_pixels

    for annotation_image, masks_dict in zip(images, all_pred_masks):
        gt_mask_flat = annotation_image.flatten()
        relevant_pixels_mask = np.zeros_like(gt_mask_flat, dtype=bool)  # Initialize as boolean

        for class_id, pred_mask in masks_dict.items():
            if class_id in label_ids:
                pred_mask_flat = pred_mask.flatten()
                if pred_mask_flat.size != gt_mask_flat.size:
                    print(f"Error: Mask size for class {class_id} is incorrect.")
                    continue

                # Ensure both masks are boolean
                gt_binary_mask = (gt_mask_flat == class_id)
                pred_binary_mask = (pred_mask_flat == 1)

                # Update relevant pixels mask
                relevant_pixels_mask |= gt_binary_mask  # Both masks are boolean, should work without error

                try:
                    iou_score = jaccard_score(gt_binary_mask, pred_binary_mask, zero_division=0)
                except ValueError as e:
                    print(f"Error calculating IoU for class {class_id}: {str(e)}")
                    iou_score = 0

                if iou_score >= 0:
                    iou_scores_per_class[class_id].append(iou_score)

                # Calculate True Positives, False Negatives for Accuracy
                tp = np.sum((gt_binary_mask == 1) & (pred_binary_mask == 1))
                fn = np.sum((gt_binary_mask == 1) & (pred_binary_mask == 0))
                acc_score = tp / (tp + fn) if (tp + fn) > 0 else 0
                acc_scores_per_class[class_id].append(acc_score)

                total_tp_sum += tp

        total_relevant_pixels += np.sum(relevant_pixels_mask)

    # Calculate final IoU and Accuracy scores
    final_iou_scores = [np.mean(iou_scores_per_class[class_id]) for class_id in label_ids if
                        iou_scores_per_class[class_id]]
    final_acc_scores = [np.mean(acc_scores_per_class[class_id]) for class_id in label_ids if
                        acc_scores_per_class[class_id]]

    mIoU = np.mean(final_iou_scores) if final_iou_scores else 0
    mAcc = np.mean(final_acc_scores) if final_acc_scores else 0
    aAcc = total_tp_sum / total_relevant_pixels if total_relevant_pixels else 0

    seg_metrics = {
        'Mean IoU (mIoU)': mIoU,
        'Mean Accuracy (mAcc)': mAcc,
        'Accuracy over all pixels (aAcc)': aAcc
    }

    return seg_metrics


def yolo_detect_and_segment(test_image_filenames, img_dir, ann_dir, categories_txt_file_path, yolov8_model_path,
                            batch_size=200, plot=False):
    """
    Process images in batches to detect and segment using YOLO, compute segmentation metrics, and optionally visualize the results.

    Args:
        test_image_filenames (list): List of filenames for test images.
        img_dir (str): Directory containing images.
        ann_dir (str): Directory containing annotation masks.
        categories_txt_file_path (str): File path for text file with category labels.
        yolov8_model_path (str): Path to the trained YOLOv8 model.
        batch_size (int, optional): Number of images to process in each batch. Default is 200.
        plot (bool, optional): If True, visualize results after processing each batch. Default is False.

    Returns:
        final_mIoU (float): Final mean Intersection over Union across all batches.
        final_mAcc (float): Final mean accuracy across all batches.
        final_aAcc (float): Final accuracy over all pixels across all batches.
    """

    print("Running detection and segmentation...")
    print(f"Images directory: {img_dir}")
    print(f"Annotation directory: {ann_dir}")
    print(f"Category file: {categories_txt_file_path}")
    print(f"Model path: {yolov8_model_path}")
    print(f"Batch size: {batch_size}")
    print(f"Plot: {plot}")
    print(f"Processing {len(test_image_filenames)} images...")

    # Retrieve class ids and class label names from tsv file
    label_names, label_ids = load_label_names_from_text_file(categories_txt_file_path)

    # Load a model
    yolo_seg_model = YOLO(yolov8_model_path)  # load an official model

    # Metrics storage
    accumulated_iou = []
    accumulated_acc = []
    accumulated_aacc = []

    # Processing in batches
    for i in range(0, len(test_image_filenames), batch_size):
        batch_filenames = test_image_filenames[i:i + batch_size]

        batch_images, batch_annotations, batch_image_paths = [], [], []
        batch_gt_bboxes = []

        for image_filename in batch_filenames:
            # Load input image and annotated masks
            original_image, annotation_image, original_image_path = load_image_and_annotation(image_filename, img_dir,
                                                                                              ann_dir)
            # Retrieve ground truth bounding boxes from annotation image
            gt_bboxes, box_label_ids = extract_bounding_boxes(annotation_image, label_ids)

            batch_images.append(original_image)
            batch_annotations.append(annotation_image)
            batch_image_paths.append(original_image_path)
            batch_gt_bboxes.append(gt_bboxes)

        # Predict with the model
        results = yolo_seg_model(batch_image_paths)

        batch_yolo_images, batch_yolo_mask_images = [], []
        batch_yolo_pred_masks, batch_yolo_pred_class_ids = [], []

        # Extract results for each image in the batch
        for j, r in enumerate(results):
            im_bgr = r.plot()  # BGR-order numpy array
            im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

            # Plot results image (only masks)
            background = np.ones_like(batch_images[j]) * 255
            yolo_mask_image = r.plot(img=background, boxes=False, labels=False, probs=False)

            # Retrieve predicted masks by YOLO
            pred_class_ids = r.boxes.cls.cpu().numpy()
            pred_masks = r.masks
            orig_shape = r.orig_shape
            processed_pred_masks = prepare_yolo_masks_for_metrics(pred_masks, pred_class_ids, orig_shape)

            batch_yolo_images.append(im_rgb)
            batch_yolo_mask_images.append(yolo_mask_image)
            batch_yolo_pred_masks.append(processed_pred_masks)
            batch_yolo_pred_class_ids.append(pred_class_ids)

        # Compute metrics for this batch
        seg_metrics = compute_segmentation_metrics(batch_annotations, batch_yolo_pred_masks, label_ids)
        accumulated_iou.append(seg_metrics['Mean IoU (mIoU)'])
        accumulated_acc.append(seg_metrics['Mean Accuracy (mAcc)'])
        accumulated_aacc.append(seg_metrics['Accuracy over all pixels (aAcc)'])

        # Optionally visualize this batch
        if plot:
            visualize_segmentation_masks(batch_images, batch_annotations, batch_yolo_images, batch_yolo_mask_images,
                                         batch_gt_bboxes, label_names)

    # Aggregated metrics after all batches
    final_mIoU = np.mean(accumulated_iou) if accumulated_iou else 0
    final_mAcc = np.mean(accumulated_acc) if accumulated_acc else 0
    final_aAcc = np.mean(accumulated_aacc) if accumulated_aacc else 0

    print(f"Final Mean IoU (mIoU): {final_mIoU}")
    print(f"Final Mean Accuracy (mAcc): {final_mAcc}")
    print(f"Final Accuracy over all pixels (aAcc): {final_aAcc}")

    return final_mIoU, final_mAcc, final_aAcc


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 Detection and Segmentation")
    parser.add_argument("img_dir", type=str, help="Directory containing the images")
    parser.add_argument("ann_dir", type=str, help="Directory containing the annotation masks")
    parser.add_argument("categories_txt_file_path", type=str,
                        help="Path to the text file containing category labels and IDs")
    parser.add_argument("yolov8_model_path", type=str, help="Path to the YOLOv8 model weights file")
    parser.add_argument("--batch_size", type=int, default=200, help="Number of images to process in each batch")
    parser.add_argument("--plot", action='store_true', help="Whether to plot the results after processing each batch")
    parser.add_argument("--image_filenames", nargs='*', default=None,
                        help="Optional list of specific image filenames to process. If not provided, all images in the directory are processed.")

    args = parser.parse_args()

    # Check if specific filenames are provided or default to retrieving all filenames from the image directory
    if args.image_filenames:
        test_image_filenames = args.image_filenames
    else:
        test_image_filenames = [f.name for f in Path(args.img_dir).glob('*')]

    # Call the function with parsed arguments
    yolo_detect_and_segment(
        test_image_filenames=test_image_filenames,
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        categories_txt_file_path=args.categories_txt_file_path,
        yolov8_model_path=args.yolov8_model_path,
        batch_size=args.batch_size,
        plot=args.plot
    )


if __name__ == "__main__":
    main()
