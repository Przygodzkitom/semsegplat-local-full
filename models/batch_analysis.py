"""
Batch Analysis Module
Quantitative analysis of segmentation predictions on uploaded images.
No ground truth comparison — pure inference + connected component analysis.
"""

import numpy as np
import cv2
from scipy.ndimage import label as connected_components
from typing import List, Dict

MIN_OBJECT_AREA = 100


def analyze_single_image(
    pred_masks: List[np.ndarray],
    class_names: List[str],
    min_object_area: int = MIN_OBJECT_AREA
) -> Dict:
    """
    Analyze prediction masks for a single image.

    Args:
        pred_masks: List of binary masks (one per class), each shape (H, W), values 0/1.
                    Output of Inferencer.predict().
        class_names: List of class name strings, length == len(pred_masks).
        min_object_area: Minimum pixel area to count an object as valid.

    Returns:
        Dictionary with per-class analysis including object counts, areas, and bounding boxes.
    """
    h, w = pred_masks[0].shape[:2]
    total_pixels = h * w
    result = {
        'image_height': h,
        'image_width': w,
        'total_pixels': total_pixels,
        'classes': {}
    }

    for i, (mask, class_name) in enumerate(zip(pred_masks, class_names)):
        if i == 0 and class_name.lower() == 'background':
            continue

        binary_mask = mask.astype(np.uint8)
        labeled_array, num_features = connected_components(binary_mask)

        objects = []
        total_area = 0

        for obj_id in range(1, num_features + 1):
            obj_mask = (labeled_array == obj_id)
            area = int(obj_mask.sum())

            if area < min_object_area:
                continue

            rows = np.where(obj_mask.any(axis=1))[0]
            cols = np.where(obj_mask.any(axis=0))[0]
            bbox = (int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1]))

            total_area += area
            objects.append({
                'id': len(objects) + 1,
                'area_pixels': area,
                'area_percent': round(area / total_pixels * 100, 4),
                'bbox': bbox
            })

        result['classes'][class_name] = {
            'object_count': len(objects),
            'total_area_pixels': total_area,
            'total_area_percent': round(total_area / total_pixels * 100, 4),
            'objects': objects
        }

    return result


def create_analysis_overlay(
    image: np.ndarray,
    pred_masks: List[np.ndarray],
    class_names: List[str],
    analysis_result: Dict,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create a visualization overlay with class masks, contours, and object labels.

    Args:
        image: Original image in RGB format, shape (H, W, 3).
        pred_masks: List of binary masks from Inferencer.predict().
        class_names: Class name strings.
        analysis_result: Output of analyze_single_image().
        alpha: Overlay transparency.

    Returns:
        Annotated image (RGB, same shape as input).
    """
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 128),
        (255, 165, 0),
    ]

    overlay = image.copy().astype(np.float32)

    color_idx = 0
    for i, (mask, class_name) in enumerate(zip(pred_masks, class_names)):
        if i == 0 and class_name.lower() == 'background':
            continue

        color = colors[color_idx % len(colors)]
        color_idx += 1

        mask_bool = mask.astype(bool)
        if mask_bool.any():
            overlay[mask_bool] = (
                overlay[mask_bool] * (1 - alpha) +
                np.array(color, dtype=np.float32) * alpha
            )

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        overlay_uint8 = overlay.astype(np.uint8)
        cv2.drawContours(overlay_uint8, contours, -1, color, 2)
        overlay = overlay_uint8.astype(np.float32)

        class_data = analysis_result['classes'].get(class_name, {})
        for obj in class_data.get('objects', []):
            bbox = obj['bbox']
            label_text = f"#{obj['id']} ({obj['area_percent']:.1f}%)"
            text_x = bbox[1]
            text_y = max(bbox[0] - 5, 15)
            cv2.putText(
                overlay_uint8, label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )
            overlay = overlay_uint8.astype(np.float32)

    return overlay.astype(np.uint8)


def aggregate_batch_results(
    all_results: List[Dict],
    filenames: List[str]
):
    """
    Aggregate per-image analysis results into a summary DataFrame.

    Args:
        all_results: List of analyze_single_image() outputs.
        filenames: Corresponding image file names.

    Returns:
        pandas DataFrame with one row per image, columns for each class's
        object_count and area_percent.
    """
    import pandas as pd

    rows = []
    for filename, result in zip(filenames, all_results):
        row = {'filename': filename}
        for class_name, class_data in result['classes'].items():
            row[f'{class_name}_count'] = class_data['object_count']
            row[f'{class_name}_area_px'] = class_data['total_area_pixels']
            row[f'{class_name}_area_pct'] = class_data['total_area_percent']
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
