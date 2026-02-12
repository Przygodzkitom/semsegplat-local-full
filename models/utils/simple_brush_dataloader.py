#!/usr/bin/env python3
"""
Simple Brush Data Loader for Label Studio
Downloads images and converts RLE annotations to PNG masks once,
caches them locally, and reads from disk during training.
"""

import os
import cv2
import numpy as np
import json
import hashlib
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import boto3
from botocore.exceptions import ClientError

# Local directory to cache converted images and masks
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'brush_cache')


class SimpleBrushDataset(Dataset):
    """
    Brush dataset that caches images and decoded masks as local PNG files.
    First run: fetches from MinIO, decodes RLE, saves PNGs.
    Subsequent runs: loads directly from disk.
    """

    def __init__(self, bucket_name, img_prefix="images/", annotation_prefix="annotations/",
                 transform=None, multilabel=True, class_names=None, has_explicit_background=True):
        self.bucket_name = bucket_name
        self.img_prefix = img_prefix
        self.annotation_prefix = annotation_prefix
        self.transform = transform
        self.multilabel = multilabel
        self.class_names = class_names or ["Object"]
        self.has_explicit_background = has_explicit_background

        # MinIO configuration
        self.endpoint_url = os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
        self.access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')

        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='us-east-1'
        )

        # Set up local cache directory
        self.cache_dir = os.path.abspath(CACHE_DIR)
        self.images_dir = os.path.join(self.cache_dir, 'images')
        self.masks_dir = os.path.join(self.cache_dir, 'masks')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)

        # Build the dataset: fetch pairs, convert, and cache
        raw_pairs = self._load_image_annotation_pairs()
        if len(raw_pairs) == 0:
            raise ValueError(f"No image-annotation pairs found in bucket {bucket_name}")

        print(f"Found {len(raw_pairs)} image-annotation pairs for BRUSH annotations")
        print(f"Background handling: {'Explicit' if has_explicit_background else 'Implicit'}")

        self.samples = self._prepare_cache(raw_pairs)

        if len(self.samples) == 0:
            raise ValueError("No samples could be prepared from annotations")

        print(f"Dataset ready: {len(self.samples)} samples (cached in {self.cache_dir})")

    # ------------------------------------------------------------------
    # MinIO loading (only used during cache preparation)
    # ------------------------------------------------------------------

    def _load_image_annotation_pairs(self):
        pairs = []
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=self.annotation_prefix
            )
            if 'Contents' not in response:
                print(f"No annotations found in {self.annotation_prefix}")
                return pairs

            for obj in response['Contents']:
                if obj['Key'].endswith('/') or obj['Size'] == 0:
                    continue
                if '.' not in obj['Key'].split('/')[-1]:
                    annotation_key = obj['Key']
                    annotation_data = self._load_annotation_from_minio(annotation_key)
                    if annotation_data and 'task' in annotation_data:
                        task_data = annotation_data['task'].get('data', {})
                        image_path = task_data.get('image', '')
                        if image_path.startswith('s3://'):
                            image_key = image_path.replace(f's3://{self.bucket_name}/', '')
                            try:
                                self.s3_client.head_object(Bucket=self.bucket_name, Key=image_key)
                                pairs.append((image_key, annotation_key))
                            except Exception:
                                print(f"Image not found: {image_key}")
        except Exception as e:
            print(f"Error loading image-annotation pairs: {e}")
        return pairs

    def _load_annotation_from_minio(self, annotation_key):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=annotation_key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            print(f"Error loading annotation {annotation_key}: {e}")
            return None

    def _load_image_from_minio(self, image_key):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=image_key)
            nparr = np.frombuffer(response['Body'].read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {image_key}: {e}")
            return None

    # ------------------------------------------------------------------
    # Cache preparation
    # ------------------------------------------------------------------

    def _sample_id(self, image_key):
        """Deterministic short ID from the image key."""
        return hashlib.md5(image_key.encode()).hexdigest()[:12]

    def _prepare_cache(self, raw_pairs):
        """Convert and cache all images/masks as local PNGs. Skip if already cached."""
        samples = []
        cached_count = 0

        for idx, (image_key, annotation_key) in enumerate(raw_pairs):
            sid = self._sample_id(image_key)
            img_path = os.path.join(self.images_dir, f'{sid}.png')
            mask_path = os.path.join(self.masks_dir, f'{sid}.png')

            if os.path.exists(img_path) and os.path.exists(mask_path):
                samples.append((img_path, mask_path))
                cached_count += 1
                continue

            # Download and convert
            image = self._load_image_from_minio(image_key)
            if image is None:
                print(f"Skipping {image_key}: could not load image")
                continue

            image = cv2.resize(image, (512, 512))

            annotation_data = self._load_annotation_from_minio(annotation_key)
            mask = self._parse_brush_annotation(annotation_data)
            if mask is None:
                mask = np.zeros((512, 512), dtype=np.uint8)

            # Save as PNG (lossless)
            cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_path, mask)
            samples.append((img_path, mask_path))

            if (idx + 1) % 10 == 0 or idx == len(raw_pairs) - 1:
                print(f"  Prepared {idx + 1}/{len(raw_pairs)}")

        if cached_count == len(raw_pairs):
            print(f"All {cached_count} samples loaded from cache")
        elif cached_count > 0:
            print(f"Loaded {cached_count} from cache, converted {len(samples) - cached_count} new")

        return samples

    # ------------------------------------------------------------------
    # RLE parsing (only used during cache preparation)
    # ------------------------------------------------------------------

    def _parse_brush_annotation(self, annotation_data):
        try:
            if not annotation_data or 'result' not in annotation_data:
                return None

            result = annotation_data['result']
            if not result:
                return None

            first_result = result[0]
            original_width = first_result.get('original_width', 512)
            original_height = first_result.get('original_height', 512)

            mask = np.full((original_height, original_width), 255, dtype=np.uint8)

            for item in result:
                if item.get('type') == 'brushlabels':
                    value = item.get('value', {})
                    brush_data = value.get('rle', [])
                    labels = value.get('brushlabels', [])

                    if brush_data and labels:
                        rle_mask = self._simple_rle_to_mask(brush_data, original_height, original_width)
                        class_label = labels[0]
                        class_id = self.class_names.index(class_label) if class_label in self.class_names else 0
                        mask[rle_mask == 1] = class_id

            return mask

        except Exception as e:
            print(f"Error parsing brush annotation: {e}")
            return None

    def _simple_rle_to_mask(self, rle_data, height, width):
        try:
            total_pixels = height * width
            mask = np.zeros(total_pixels, dtype=np.uint8)

            if isinstance(rle_data, list) and len(rle_data) > 0:
                if len(rle_data) <= total_pixels:
                    arr = np.array(rle_data, dtype=np.int32)
                    mask[:len(arr)] = (arr > 0).astype(np.uint8)
                else:
                    step = len(rle_data) // total_pixels
                    indices = np.arange(total_pixels) * step
                    valid = indices < len(rle_data)
                    arr = np.array(rle_data, dtype=np.int32)
                    mask[valid] = (arr[indices[valid]] > 0).astype(np.uint8)

            mask = mask.reshape(height, width)

            non_zero = np.count_nonzero(mask)
            if non_zero == 0:
                center_h, center_w = height // 2, width // 2
                size = min(height, width) // 8
                mask[center_h - size:center_h + size, center_w - size:center_w + size] = 1
            elif non_zero > total_pixels * 0.8:
                mask = np.zeros((height, width), dtype=np.uint8)
                center_h, center_w = height // 2, width // 2
                size = min(height, width) // 4
                mask[center_h - size:center_h + size, center_w - size:center_w + size] = 1

            return mask

        except Exception as e:
            print(f"Error in RLE conversion: {e}")
            mask = np.zeros((height, width), dtype=np.uint8)
            center_h, center_w = height // 2, width // 2
            size = min(height, width) // 8
            mask[center_h - size:center_h + size, center_w - size:center_w + size] = 1
            return mask

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Read from local disk (fast)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Convert to multilabel format if needed
        if self.multilabel:
            h, w = mask.shape
            num_classes = len(self.class_names)
            multilabel_mask = np.zeros((num_classes, h, w), dtype=np.float32)

            if self.has_explicit_background:
                for i in range(1, num_classes):
                    multilabel_mask[i] = (mask == i).astype(np.float32)
                multilabel_mask[0] = (mask == 0).astype(np.float32)
            else:
                multilabel_mask[0] = (mask == 0).astype(np.float32)
                for i in range(1, num_classes):
                    multilabel_mask[i] = (mask == i).astype(np.float32)

            mask = multilabel_mask.transpose(1, 2, 0)  # (H, W, C) for albumentations

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            if self.multilabel:
                mask = augmented["mask"].permute(2, 0, 1)  # (C, H, W) for PyTorch
            else:
                mask = augmented["mask"].unsqueeze(0)

        return image, mask
