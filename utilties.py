
# IMPORTS
import os
import cv2
import numpy as np


def load_image_and_annotation(image_filename, image_dir, ann_dir):
    """
    Load an image and its corresponding annotation mask from specified directories.

    Args:
        image_filename (str): Filename of the input image.
        image_dir (str): Directory where images are stored.
        ann_dir (str): Directory where annotation masks are stored.

    Returns:
        original_image (np.ndarray): The loaded image.
        annotation_image (np.ndarray): The corresponding annotation mask.
        original_image_path (str): Path to the loaded image.
    """
    # Load the original image
    original_image_path = os.path.join(image_dir, image_filename)
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Load the annotation mask
    image_id = os.path.splitext(image_filename)[0]
    annotation_path = os.path.join(ann_dir, f"{image_id}.png")
    annotation_image = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)

    return original_image, annotation_image, original_image_path


def load_label_names_from_text_file(text_file_path):
    """
    Load label names from a text file.
    Each line in the text file should contain a label ID and its corresponding name, separated by a tab.

    Args:
        text_file_path (str): Path to the text file containing label IDs and names.

    Returns:
        label_names (dict): Dictionary mapping label IDs (int) to their corresponding names (str).
        label_ids (list): List of all label IDs.
    """
    label_names = {}
    label_ids = []
    with open(text_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Assuming a single tab or consistent spacing separates the label ID and name
            parts = line.strip().split('\t')  # Change '\t' to ' ' if spaces are used
            if len(parts) == 2:
                label_id = int(parts[0])
                label_name = parts[1]
                label_names[label_id] = label_name
                label_ids.append(label_id)
    return label_names, label_ids


def class_id_to_color(class_id):
    """
    Generate a consistent RGB color for a given class ID using a seed based on the class ID.

    Args:
        class_id (int): The class ID for which a color is generated.

    Returns:
        ndarray: An array representing the RGB color.
    """
    # Generate a consistent color based on the class ID
    np.random.seed(class_id)
    color = np.random.rand(3,)
    return color


def show_mask(mask, class_id, ax, alpha=0.7):
    """
    Overlay a mask on a plot using a color specific to the class ID.

    Args:
        mask (np.ndarray): The binary mask to be displayed.
        class_id (int): The class ID used to determine the color of the mask.
        ax (matplotlib.axes.Axes): The plot axis on which to overlay the mask.
        alpha (float, optional): The opacity level of the mask. Defaults to 0.7.
    """
    # Get the color for the class
    color = class_id_to_color(class_id)

    # Create an RGBA mask with the specified color and alpha
    rgba_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    rgba_mask[..., :3] = color  # RGB channels
    rgba_mask[..., 3] = (mask > 0) * alpha  # Alpha channel

    # Overlay mask on the existing plot
    ax.imshow(rgba_mask, interpolation='nearest')
