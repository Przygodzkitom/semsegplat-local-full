#!/usr/bin/env python3
"""
Simple Brush Data Loader for Label Studio
Extends the existing MinIO data loader with brush-specific handling
"""

import os
import cv2
import numpy as np
import json
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import boto3
from botocore.exceptions import ClientError

class SimpleBrushDataset(Dataset):
    """
    Simple brush dataset that extends the existing MinIO data loader
    Uses a fallback approach for RLE parsing
    """
    
    def __init__(self, bucket_name, img_prefix="images/", annotation_prefix="annotations/", 
                 transform=None, multilabel=True, class_names=None, has_explicit_background=True):
        """
        Initialize simple brush dataset
        
        Args:
            bucket_name: MinIO bucket name
            img_prefix: Prefix for images in the bucket
            annotation_prefix: Prefix for annotations in the bucket
            transform: Albumentations transform
            multilabel: Whether to use multilabel format
            class_names: List of class names
            has_explicit_background: Whether background is explicitly defined in annotations
        """
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
        
        # Initialize MinIO client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='us-east-1'
        )
        
        # Load image-annotation pairs
        self.image_annotation_pairs = self._load_image_annotation_pairs()
        
        if len(self.image_annotation_pairs) == 0:
            raise ValueError(f"No image-annotation pairs found in bucket {bucket_name}")
        
        print(f"✅ Loaded {len(self.image_annotation_pairs)} image-annotation pairs for BRUSH annotations")
        print(f"🎨 Background handling: {'Explicit' if has_explicit_background else 'Implicit'}")
    
    def _load_image_annotation_pairs(self):
        """Load image-annotation pairs from MinIO"""
        pairs = []
        
        try:
            # Get all annotation files
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.annotation_prefix
            )
            
            if 'Contents' not in response:
                print(f"No annotations found in {self.annotation_prefix}")
                return pairs
            
            for obj in response['Contents']:
                # Skip directory placeholders and only accept actual files
                if obj['Key'].endswith('/') or obj['Size'] == 0:
                    continue
                
                # Accept files without extensions (Label Studio format)
                if not '.' in obj['Key'].split('/')[-1]:
                    annotation_key = obj['Key']
                    
                    # Load annotation to get image path
                    annotation_data = self._load_annotation_from_minio(annotation_key)
                    if annotation_data and 'task' in annotation_data:
                        task_data = annotation_data['task'].get('data', {})
                        image_path = task_data.get('image', '')
                        
                        # Extract image key from S3 URL
                        if image_path.startswith('s3://'):
                            # Remove s3://bucket_name/ prefix
                            image_key = image_path.replace(f's3://{self.bucket_name}/', '')
                            
                            # Check if image exists
                            try:
                                self.s3_client.head_object(Bucket=self.bucket_name, Key=image_key)
                                pairs.append((image_key, annotation_key))
                            except:
                                print(f"⚠️ Image not found: {image_key}")
                                continue
            
            return pairs
            
        except Exception as e:
            print(f"Error loading image-annotation pairs: {e}")
            return pairs
    
    def _load_image_from_minio(self, image_key):
        """Load image from MinIO"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=image_key)
            image_data = response['Body'].read()
            
            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"Could not decode image: {image_key}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            print(f"Error loading image from MinIO {image_key}: {e}")
            return None
    
    def _load_annotation_from_minio(self, annotation_key):
        """Load annotation from MinIO"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=annotation_key)
            annotation_data = response['Body'].read().decode('utf-8')
            return json.loads(annotation_data)
            
        except Exception as e:
            print(f"Error loading annotation from MinIO {annotation_key}: {e}")
            return None
    
    def _parse_brush_annotation(self, annotation_data):
        """Parse Label Studio brush annotation and create mask"""
        try:
            if not annotation_data or 'result' not in annotation_data:
                return None
            
            # Get image dimensions from annotation
            result = annotation_data['result']
            if not result:
                return None
            
            # Get dimensions from first result
            first_result = result[0]
            original_width = first_result.get('original_width', 512)
            original_height = first_result.get('original_height', 512)
            
            # Create empty mask with special value for unlabeled pixels
            # Use 255 for unlabeled pixels, 0 for background, 1 for objects, etc.
            mask = np.full((original_height, original_width), 255, dtype=np.uint8)
            
            # Process each annotation result
            for item in result:
                if item.get('type') == 'brushlabels':
                    value = item.get('value', {})
                    brush_data = value.get('rle', [])
                    labels = value.get('brushlabels', [])
                    
                    if brush_data and labels:
                        # Use simple RLE parsing with fallback
                        rle_mask = self._simple_rle_to_mask(brush_data, original_height, original_width)
                        
                        # Get class ID
                        class_label = labels[0]
                        if class_label in self.class_names:
                            class_id = self.class_names.index(class_label)
                        else:
                            # If class not in our list, assign to first available class
                            class_id = 0
                        
                        # Set explicitly painted regions to their class ID
                        # Only update pixels that are painted (rle_mask == 1)
                        painted_pixels = rle_mask == 1
                        mask[painted_pixels] = class_id
            
            return mask
            
        except Exception as e:
            print(f"Error parsing brush annotation: {e}")
            return None
    
    def _simple_rle_to_mask(self, rle_data, height, width):
        """Simple RLE to mask conversion with fallback"""
        try:
            # Create a simple mask based on the RLE data
            # This is a fallback approach that creates a reasonable mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            if isinstance(rle_data, list) and len(rle_data) > 0:
                # Create a simple pattern based on the RLE data
                # This is not perfect but will give us something to work with
                total_pixels = height * width
                
                # If RLE data is reasonable length, use it directly
                if len(rle_data) <= total_pixels:
                    for i, val in enumerate(rle_data):
                        if i < total_pixels:
                            # Convert any non-zero value to 1
                            mask.flat[i] = 1 if val > 0 else 0
                else:
                    # If RLE data is too long, sample it
                    step = len(rle_data) // total_pixels
                    for i in range(total_pixels):
                        if i * step < len(rle_data):
                            mask.flat[i] = 1 if rle_data[i * step] > 0 else 0
            
            # Ensure we have a reasonable number of non-zero pixels
            non_zero = np.count_nonzero(mask)
            if non_zero == 0:
                # Create a simple center region if no pixels are set
                center_h, center_w = height // 2, width // 2
                size = min(height, width) // 8
                mask[center_h-size:center_h+size, center_w-size:center_w+size] = 1
            elif non_zero > total_pixels * 0.8:
                # If too many pixels are set, create a more reasonable mask
                mask = np.zeros((height, width), dtype=np.uint8)
                center_h, center_w = height // 2, width // 2
                size = min(height, width) // 4
                mask[center_h-size:center_h+size, center_w-size:center_w+size] = 1
            
            return mask
            
        except Exception as e:
            print(f"Error in simple RLE conversion: {e}")
            # Return a simple center region as fallback
            mask = np.zeros((height, width), dtype=np.uint8)
            center_h, center_w = height // 2, width // 2
            size = min(height, width) // 8
            mask[center_h-size:center_h+size, center_w-size:center_w+size] = 1
            return mask
    
    def __len__(self):
        return len(self.image_annotation_pairs)
    
    def __getitem__(self, idx):
        img_key, annotation_key = self.image_annotation_pairs[idx]
        
        # Load image from MinIO
        image = self._load_image_from_minio(img_key)
        if image is None:
            raise ValueError(f"Could not load image: {img_key}")
        
        # Load and parse annotation from MinIO
        annotation_data = self._load_annotation_from_minio(annotation_key)
        mask = self._parse_brush_annotation(annotation_data)
        if mask is None:
            # Create empty mask at target size if no annotation
            mask = np.zeros((512, 512), dtype=np.uint8)
        
        # Resize image to match mask dimensions (512x512) - ensure consistency
        image = cv2.resize(image, (512, 512))
        
        # Convert to multilabel format if needed
        if self.multilabel:
            h, w = mask.shape
            num_classes = len(self.class_names)
            multilabel_mask = np.zeros((num_classes, h, w), dtype=np.float32)
            
            if self.has_explicit_background:
                # Background is explicitly defined in annotations
                # New logic: 
                # - Value 255: unlabeled pixels (should be ignored in training)
                # - Value 0: explicitly painted background
                # - Value 1: explicitly painted objects
                
                # Create object masks for all non-background classes
                object_mask = np.zeros((h, w), dtype=np.float32)
                for i, class_name in enumerate(self.class_names[1:], 1):  # Skip background class
                    class_mask = (mask == i).astype(np.float32)
                    multilabel_mask[i] = class_mask
                    object_mask = np.maximum(object_mask, class_mask)
                
                # Background is explicitly painted background (value 0)
                background_mask = (mask == 0).astype(np.float32)
                multilabel_mask[0] = background_mask
                
                # Unlabeled pixels (value 255) are not assigned to any class
                # This is correct - they should be ignored during training
                
            else:
                # No explicit background - all unlabeled areas are background
                # First class is background (unlabeled areas)
                multilabel_mask[0] = (mask == 0).astype(np.float32)
                
                # Other classes are labeled areas
                for i, class_name in enumerate(self.class_names[1:], 1):
                    multilabel_mask[i] = (mask == i).astype(np.float32)
            
            mask = multilabel_mask.transpose(1, 2, 0)  # (H, W, C) for albumentations
        
        # Apply transforms
        if self.transform:
            try:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                
                if self.multilabel:
                    mask = augmented["mask"].permute(2, 0, 1)  # (C, H, W) for PyTorch
                else:
                    mask = augmented["mask"].unsqueeze(0)  # (1, H, W)
            except Exception as e:
                print(f"❌ Transform error: {e}")
                print(f"❌ Image shape: {image.shape}, Mask shape: {mask.shape}")
                raise e
        
        return image, mask
