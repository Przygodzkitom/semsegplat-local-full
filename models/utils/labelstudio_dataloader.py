import os
import cv2
import numpy as np
import json
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from google.cloud import storage

class LabelStudioDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None, multilabel=True, class_names=None):
        """
        Dataset for LabelStudio annotations
        
        Args:
            img_dir: Directory containing images
            annotation_dir: Directory containing LabelStudio JSON annotation files
            transform: Albumentations transforms
            multilabel: Whether to use multilabel format (2 channels for 2 classes)
            class_names: List of class names in order [background, object]
        """
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.multilabel = multilabel
        self.class_names = class_names or ["Background", "Object"]
        
        # Get all image files
        self.image_files = [f for f in os.listdir(img_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Get all annotation files
        self.annotation_files = [f for f in os.listdir(annotation_dir) 
                                if f.endswith('.json')]
        
        # Create mapping between images and annotations
        self.image_annotation_pairs = []
        for img_file in self.image_files:
            # Find corresponding annotation file
            img_name = os.path.splitext(img_file)[0]
            annotation_file = f"{img_name}.json"
            
            if annotation_file in self.annotation_files:
                self.image_annotation_pairs.append((img_file, annotation_file))
        
        print(f"Found {len(self.image_annotation_pairs)} image-annotation pairs")
        
    def __len__(self):
        return len(self.image_annotation_pairs)
    
    def _load_image_from_gcs(self, image_url):
        """Load image from GCS URL"""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error loading image from {image_url}: {e}")
            return None
    
    def _parse_labelstudio_annotation(self, annotation_file):
        """Parse LabelStudio JSON annotation and convert to mask"""
        try:
            with open(os.path.join(self.annotation_dir, annotation_file), 'r') as f:
                annotation_data = json.load(f)
            
            # Get the first annotation (assuming one annotation per image)
            if not annotation_data.get('annotations'):
                return None
            
            annotation = annotation_data['annotations'][0]
            result = annotation.get('result', [])
            
            # Get image dimensions from the annotation
            image_width = annotation_data.get('data', {}).get('image', {}).get('width', 512)
            image_height = annotation_data.get('data', {}).get('image', {}).get('height', 512)
            
            # Create empty mask
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            
            # Process polygon annotations
            for item in result:
                if item.get('type') == 'polygonlabels':
                    value = item.get('value', {})
                    points = value.get('points', [])
                    
                    if points:
                        # Convert points to polygon coordinates
                        polygon_points = []
                        for i in range(0, len(points), 2):
                            x = int(points[i] * image_width / 100)  # Convert percentage to pixels
                            y = int(points[i + 1] * image_height / 100)
                            polygon_points.append([x, y])
                        
                        # Get class label
                        labels = value.get('polygonlabels', [])
                        if labels:
                            class_label = labels[0]
                            class_id = self.class_names.index(class_label) if class_label in self.class_names else 1
                            
                            # Create polygon mask
                            polygon_mask = self._polygon_to_mask(polygon_points, image_height, image_width)
                            
                            # Add to main mask
                            mask = np.maximum(mask, polygon_mask * class_id)
                
                elif item.get('type') == 'brushlabels':
                    # Handle brush annotations if they exist
                    value = item.get('value', {})
                    brush_data = value.get('rle', [])
                    
                    if brush_data:
                        # Convert RLE to mask
                        rle_mask = self._rle_to_mask(brush_data, image_height, image_width)
                        
                        # Get class label
                        labels = value.get('brushlabels', [])
                        if labels:
                            class_label = labels[0]
                            class_id = self.class_names.index(class_label) if class_label in self.class_names else 1
                            
                            # Add to mask
                            mask = np.maximum(mask, rle_mask * class_id)
            
            return mask
            
        except Exception as e:
            print(f"Error parsing annotation {annotation_file}: {e}")
            return None
    
    def _polygon_to_mask(self, polygon_points, height, width):
        """Convert polygon points to binary mask"""
        try:
            from PIL import Image, ImageDraw
            
            # Create a PIL image for drawing
            img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img)
            
            # Draw the polygon
            if len(polygon_points) >= 3:  # Need at least 3 points for a polygon
                # polygon_points should already be tuples of integers from _parse_labelstudio_annotation
                draw.polygon(polygon_points, fill=1)
            
            # Convert to numpy array
            mask = np.array(img)
            return mask
            
        except Exception as e:
            print(f"Error converting polygon to mask: {e}")
            return np.zeros((height, width), dtype=np.uint8)
    
    def _rle_to_mask(self, rle_data, height, width):
        """Convert RLE (Run Length Encoding) to binary mask"""
        try:
            # LabelStudio RLE format: [start, length, start, length, ...]
            mask = np.zeros(height * width, dtype=np.uint8)
            
            if len(rle_data) % 2 == 0:
                # Even length - process normally
                for i in range(0, len(rle_data), 2):
                    start = rle_data[i]
                    length = rle_data[i + 1]
                    if start < len(mask) and start + length <= len(mask) and length > 0:
                        mask[start:start + length] = 1
            else:
                # Odd length - truncate to even length
                print(f"⚠️ DEBUG: Odd-length RLE data ({len(rle_data)}), truncating to even length")
                even_length = len(rle_data) - 1
                for i in range(0, even_length, 2):
                    start = rle_data[i]
                    length = rle_data[i + 1]
                    if start < len(mask) and start + length <= len(mask) and length > 0:
                        mask[start:start + length] = 1
            
            return mask.reshape(height, width)
            
        except Exception as e:
            print(f"Error converting RLE to mask: {e}")
            return np.zeros((height, width), dtype=np.uint8)
    
    def __getitem__(self, idx):
        img_file, annotation_file = self.image_annotation_pairs[idx]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            # Try loading from GCS if local file doesn't exist
            # This would need to be implemented based on your GCS setup
            raise ValueError(f"Could not load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load and parse annotation
        mask = self._parse_labelstudio_annotation(annotation_file)
        if mask is None:
            # Create empty mask if no annotation
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Convert to multilabel format if needed
        if self.multilabel:
            h, w = mask.shape
            num_classes = len(self.class_names)
            multilabel_mask = np.zeros((num_classes, h, w), dtype=np.float32)
            
            # Create binary masks for each class
            for i, class_name in enumerate(self.class_names):
                if i == 0:  # Background class
                    multilabel_mask[i] = (mask == 0).astype(np.float32)
                else:  # Object classes
                    multilabel_mask[i] = (mask == i).astype(np.float32)
            
            mask = multilabel_mask.transpose(1, 2, 0)  # (H, W, C) for albumentations
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            
            if self.multilabel:
                mask = augmented["mask"].permute(2, 0, 1)  # (C, H, W) for PyTorch
            else:
                mask = augmented["mask"].unsqueeze(0)  # (1, H, W)
        
        return image, mask


class PlateletDataset(Dataset):
    """Legacy dataset for pixel-numbered masks (compatibility with training.py)"""
    def __init__(self, img_dir, mask_dir, transform=None, multilabel=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.multilabel = multilabel
        
        self.image_files = sorted([f for f in os.listdir(img_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (assumes grayscale with pixel values 0, 1, 2)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if self.multilabel:
            # Convert [0, 1, 2] → [2, H, W] binary mask
            h, w = mask_raw.shape
            mask = np.zeros((2, h, w), dtype=np.float32)
            mask[0] = (mask_raw == 1).astype(np.float32)  # Class 1
            mask[1] = (mask_raw == 2).astype(np.float32)  # Class 2
            mask = mask.transpose(1, 2, 0)  # albumentations expects (H, W, C)
        else:
            mask = mask_raw.astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            
            if self.multilabel:
                mask = augmented["mask"].permute(2, 0, 1)  # (C, H, W) float
            else:
                mask = augmented["mask"].unsqueeze(0)  # (1, H, W)
        
        return image, mask 

class LabelStudioGCSDataset(Dataset):
    """Dataset for LabelStudio annotations stored in GCS"""
    def __init__(self, bucket_name, img_prefix="images/", annotation_prefix="annotations/", 
                 transform=None, multilabel=True, class_names=None):
        """
        Dataset for LabelStudio annotations from GCS
        
        Args:
            bucket_name: GCS bucket name
            img_prefix: Prefix for images in the bucket
            annotation_prefix: Prefix for annotations in the bucket
            transform: Albumentations transforms
            multilabel: Whether to use multilabel format (2 channels for 2 classes)
            class_names: List of class names in order [background, object]
        """
        self.bucket_name = bucket_name
        self.img_prefix = img_prefix
        self.annotation_prefix = annotation_prefix
        self.transform = transform
        self.multilabel = multilabel
        self.class_names = class_names or ["Background", "Object"]
        
        # Initialize GCS client
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Get all image files from GCS
        self.image_files = []
        blobs = self.bucket.list_blobs(prefix=img_prefix)
        for blob in blobs:
            if blob.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                self.image_files.append(blob.name)
        
        # Get all annotation files from GCS
        self.annotation_files = []
        blobs = self.bucket.list_blobs(prefix=annotation_prefix)
        for blob in blobs:
            # Skip directory placeholders and only accept actual files
            if blob.name.endswith('/') or blob.size == 0:
                continue
            # Accept numeric annotation files (e.g., masks/1, masks/10, etc.)
            if blob.name.split('/')[-1].isdigit():
                self.annotation_files.append(blob.name)
            # Also accept .json files if present (for future compatibility)
            elif blob.name.endswith('.json'):
                self.annotation_files.append(blob.name)
        
        # Create mapping between images and annotations
        self.image_annotation_pairs = []
        debug_printed = 0
        for img_blob_name in self.image_files:
            img_name = os.path.basename(img_blob_name)
            img_name_without_ext = os.path.splitext(img_name)[0]
            for annotation_blob_name in self.annotation_files:
                try:
                    annotation_data = self._load_annotation_from_gcs(annotation_blob_name)
                    if annotation_data:
                        # Use the correct path for image URL
                        image_url = annotation_data.get('task', {}).get('data', {}).get('image', '')
                        if debug_printed < 5:
                            print(f"[DEBUG] Image: {img_name} | Annotation: {annotation_blob_name} | image_url in JSON: {image_url}")
                            debug_printed += 1
                        if image_url:
                            if image_url.startswith('gs://'):
                                url_filename = os.path.basename(image_url)
                            elif '/' in image_url:
                                url_filename = os.path.basename(image_url)
                            else:
                                url_filename = image_url
                            url_name_without_ext = os.path.splitext(url_filename)[0]
                            if url_name_without_ext == img_name_without_ext:
                                self.image_annotation_pairs.append((img_blob_name, annotation_blob_name))
                                break
                except Exception as e:
                    print(f"Error checking annotation {annotation_blob_name}: {e}")
                    continue
        
        print(f"Found {len(self.image_annotation_pairs)} image-annotation pairs in GCS")
        
    def __len__(self):
        return len(self.image_annotation_pairs)
    
    def _load_image_from_gcs(self, blob_name):
        """Load image from GCS"""
        try:
            blob = self.bucket.blob(blob_name)
            image_data = blob.download_as_bytes()
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image from GCS {blob_name}: {e}")
            return None
    
    def _load_annotation_from_gcs(self, blob_name):
        """Load annotation from GCS"""
        try:
            blob = self.bucket.blob(blob_name)
            annotation_data = blob.download_as_text()
            return json.loads(annotation_data)
        except Exception as e:
            print(f"Error loading annotation from GCS {blob_name}: {e}")
            return None
    
    def _parse_labelstudio_annotation(self, annotation_data):
        """Parse LabelStudio annotation JSON and convert to mask"""
        try:
            result = annotation_data.get('result', [])
            if not result:
                return None
            
            # Get image dimensions from the first annotation
            first_item = result[0]
            image_width = first_item.get('original_width', 512)
            image_height = first_item.get('original_height', 512)
            
            # Create empty mask
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            
            # Process polygon annotations
            for item in result:
                if item.get('type') == 'polygonlabels':
                    value = item.get('value', {})
                    points = value.get('points', [])
                    
                    if points:
                        # Convert points to polygon coordinates
                        # Points are already in [x, y] format from LabelStudio
                        polygon_points = []
                        for point in points:
                            if len(point) == 2:
                                x = int(point[0] * image_width / 100)  # Convert percentage to pixels
                                y = int(point[1] * image_height / 100)
                                polygon_points.append([x, y])
                        
                        # Get class label
                        labels = value.get('polygonlabels', [])
                        if labels:
                            class_label = labels[0]
                            class_id = self.class_names.index(class_label) if class_label in self.class_names else 1
                            
                            # Create polygon mask
                            polygon_mask = self._polygon_to_mask(polygon_points, image_height, image_width)
                            
                            # Add to main mask
                            mask = np.maximum(mask, polygon_mask * class_id)
                
                elif item.get('type') == 'brushlabels':
                    # Handle brush annotations if they exist
                    value = item.get('value', {})
                    brush_data = value.get('rle', [])
                    
                    if brush_data:
                        # Convert RLE to mask
                        rle_mask = self._rle_to_mask(brush_data, image_height, image_width)
                        
                        # Get class label
                        labels = value.get('brushlabels', [])
                        if labels:
                            class_label = labels[0]
                            class_id = self.class_names.index(class_label) if class_label in self.class_names else 1
                            
                            # Add to mask
                            mask = np.maximum(mask, rle_mask * class_id)
            
            return mask
            
        except Exception as e:
            print(f"Error parsing annotation: {e}")
            return None
    
    def _polygon_to_mask(self, polygon_points, height, width):
        """Convert polygon points to binary mask"""
        try:
            from PIL import Image, ImageDraw
            
            # Create a PIL image for drawing
            img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img)
            
            # Draw the polygon
            if len(polygon_points) >= 3:  # Need at least 3 points for a polygon
                draw.polygon(polygon_points, fill=1)
            
            # Convert to numpy array
            mask = np.array(img)
            return mask
            
        except Exception as e:
            print(f"Error converting polygon to mask: {e}")
            return np.zeros((height, width), dtype=np.uint8)
    
    def _rle_to_mask(self, rle_data, height, width):
        """Convert RLE (Run Length Encoding) to binary mask"""
        try:
            # LabelStudio RLE format: [start, length, start, length, ...]
            mask = np.zeros(height * width, dtype=np.uint8)
            
            if len(rle_data) % 2 == 0:
                # Even length - process normally
                for i in range(0, len(rle_data), 2):
                    start = rle_data[i]
                    length = rle_data[i + 1]
                    if start < len(mask) and start + length <= len(mask) and length > 0:
                        mask[start:start + length] = 1
            else:
                # Odd length - truncate to even length
                print(f"⚠️ DEBUG: Odd-length RLE data ({len(rle_data)}), truncating to even length")
                even_length = len(rle_data) - 1
                for i in range(0, even_length, 2):
                    start = rle_data[i]
                    length = rle_data[i + 1]
                    if start < len(mask) and start + length <= len(mask) and length > 0:
                        mask[start:start + length] = 1
            
            return mask.reshape(height, width)
            
        except Exception as e:
            print(f"Error converting RLE to mask: {e}")
            return np.zeros((height, width), dtype=np.uint8)
    
    def __getitem__(self, idx):
        img_blob_name, annotation_blob_name = self.image_annotation_pairs[idx]
        
        # Load image from GCS
        image = self._load_image_from_gcs(img_blob_name)
        if image is None:
            raise ValueError(f"Could not load image: {img_blob_name}")
        
        # Load and parse annotation from GCS
        annotation_data = self._load_annotation_from_gcs(annotation_blob_name)
        mask = self._parse_labelstudio_annotation(annotation_data)
        if mask is None:
            # Create empty mask if no annotation
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Convert to multilabel format if needed
        if self.multilabel:
            h, w = mask.shape
            multilabel_mask = np.zeros((2, h, w), dtype=np.float32)
            multilabel_mask[0] = (mask == 0).astype(np.float32)  # Background
            multilabel_mask[1] = (mask == 1).astype(np.float32)  # Object
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

class LabelStudioGCSDatasetNumeric(Dataset):
    """Dataset for LabelStudio annotations stored in GCS with numeric mask names"""
    def __init__(self, bucket_name, img_prefix="images/", annotation_prefix="masks/", 
                 transform=None, multilabel=True, class_names=None):
        """
        Dataset for LabelStudio annotations from GCS with numeric mask names
        
        Args:
            bucket_name: GCS bucket name
            img_prefix: Prefix for images in the bucket
            annotation_prefix: Prefix for annotations in the bucket
            transform: Albumentations transforms
            multilabel: Whether to use multilabel format (2 channels for 2 classes)
            class_names: List of class names in order [background, object]
        """
        self.bucket_name = bucket_name
        self.img_prefix = img_prefix
        self.annotation_prefix = annotation_prefix
        self.transform = transform
        self.multilabel = multilabel
        self.class_names = class_names or ["Background", "Object"]
        
        # Initialize GCS client
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Get all image files from GCS
        self.image_files = []
        blobs = self.bucket.list_blobs(prefix=img_prefix)
        for blob in blobs:
            if blob.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                self.image_files.append(blob.name)
        
        # Get all annotation files from GCS (numeric names)
        self.annotation_files = []
        blobs = self.bucket.list_blobs(prefix=annotation_prefix)
        print(f"🔍 Found {len(list(blobs))} total blobs in annotation prefix '{annotation_prefix}'")
        
        # Reset the iterator
        blobs = self.bucket.list_blobs(prefix=annotation_prefix)
        for blob in blobs:
            print(f"🔍 Checking blob: {blob.name} (size: {blob.size})")
            # Skip directory placeholders and only accept actual files
            if blob.name.endswith('/') or blob.size == 0:
                print(f"🔍 Skipping directory/empty blob: {blob.name}")
                continue
            # Accept numeric annotation files (e.g., masks/1, masks/10, etc.)
            filename = blob.name.split('/')[-1]
            print(f"🔍 Filename: '{filename}', isdigit: {filename.isdigit()}")
            if filename.isdigit():
                self.annotation_files.append(blob.name)
                print(f"✅ Added annotation file: {blob.name}")
            else:
                print(f"🔍 Skipping non-numeric file: {blob.name}")
        
        print(f"🔍 Total annotation files found: {len(self.annotation_files)}")
        print(f"🔍 Annotation files: {self.annotation_files[:10]}")  # Show first 10
        
        # Create mapping between images and annotations by reading JSON content
        self.image_annotation_pairs = []
        
        print(f"🔍 Matching {len(self.image_files)} images with {len(self.annotation_files)} annotations...")
        print(f"🔍 Image prefix: '{img_prefix}'")
        print(f"🔍 Annotation prefix: '{annotation_prefix}'")
        
        for annotation_blob_name in self.annotation_files:
            try:
                print(f"🔍 Processing annotation: {annotation_blob_name}")
                
                # Load annotation from GCS
                annotation_data = self._load_annotation_from_gcs(annotation_blob_name)
                if annotation_data:
                    print(f"🔍 Annotation data keys: {list(annotation_data.keys())}")
                    
                    # Extract image reference from JSON
                    image_url = annotation_data.get('task', {}).get('data', {}).get('image', '')
                    print(f"🔍 Extracted image URL: '{image_url}'")
                    
                    if image_url:
                        # Convert GCS URL to blob name
                        if image_url.startswith('gs://'):
                            # Remove gs://bucket-name/ prefix
                            image_blob_name = image_url.replace(f'gs://{bucket_name}/', '')
                        else:
                            image_blob_name = image_url
                        
                        print(f"🔍 Looking for image blob: '{image_blob_name}'")
                        print(f"🔍 Available images (first 5): {self.image_files[:5]}")
                        
                        # Check if this image exists in our image list
                        if image_blob_name in self.image_files:
                            self.image_annotation_pairs.append((image_blob_name, annotation_blob_name))
                            print(f"✅ Matched: {os.path.basename(image_blob_name)} -> {os.path.basename(annotation_blob_name)}")
                        else:
                            print(f"❌ Image not found: {image_blob_name}")
                    else:
                        print(f"❌ No image reference in annotation: {annotation_blob_name}")
                        print(f"🔍 Full annotation structure: {json.dumps(annotation_data, indent=2)[:500]}...")
                        
            except Exception as e:
                print(f"❌ Error processing annotation {annotation_blob_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Found {len(self.image_annotation_pairs)} image-annotation pairs in GCS")
        print(f"Images: {len(self.image_files)}, Annotations: {len(self.annotation_files)}")
        
        # Show first few pairs for debugging
        for i, (img, ann) in enumerate(self.image_annotation_pairs[:3]):
            print(f"Pair {i+1}: {os.path.basename(img)} -> {os.path.basename(ann)}")
        
    def __len__(self):
        return len(self.image_annotation_pairs)
    
    def _load_image_from_gcs(self, blob_name):
        """Load image from GCS"""
        try:
            blob = self.bucket.blob(blob_name)
            image_data = blob.download_as_bytes()
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image from GCS {blob_name}: {e}")
            return None
    
    def _load_annotation_from_gcs(self, blob_name):
        """Load annotation from GCS"""
        try:
            blob = self.bucket.blob(blob_name)
            annotation_data = blob.download_as_text()
            return json.loads(annotation_data)
        except Exception as e:
            print(f"Error loading annotation from GCS {blob_name}: {e}")
            return None
    
    def _parse_labelstudio_annotation(self, annotation_data):
        """Parse LabelStudio JSON annotation and convert to mask"""
        try:
            # Get the result from the annotation
            result = annotation_data.get('result', [])
            
            # Use fixed target size for mask (matching the resize transform)
            target_width = 512
            target_height = 512
            
            # Create empty mask at target size
            mask = np.zeros((target_height, target_width), dtype=np.uint8)
            
            # Process polygon annotations
            for item in result:
                if item.get('type') == 'polygonlabels':
                    value = item.get('value', {})
                    points = value.get('points', [])
                    
                    if points:
                        # Convert points to polygon coordinates at target size
                        # Points are already in [x, y] format from LabelStudio
                        polygon_points = []
                        for point in points:
                            if len(point) == 2:
                                # Convert percentage to pixels at target size
                                x = int(point[0] * target_width / 100)
                                y = int(point[1] * target_height / 100)
                                polygon_points.append((x, y))  # Convert to tuple for PIL
                        
                        # Get class label
                        labels = value.get('polygonlabels', [])
                        if labels:
                            class_label = labels[0]
                            class_id = self.class_names.index(class_label) if class_label in self.class_names else 1
                            
                            # Convert polygon to mask at target size
                            polygon_mask = self._polygon_to_mask(polygon_points, target_height, target_width)
                            
                            # Add to main mask
                            mask = np.maximum(mask, polygon_mask * class_id)
                
                elif item.get('type') == 'brushlabels':
                    # Handle brush annotations if they exist
                    value = item.get('value', {})
                    brush_data = value.get('rle', [])
                    
                    if brush_data:
                        # Convert RLE to mask at target size
                        rle_mask = self._rle_to_mask(brush_data, target_height, target_width)
                        
                        # Get class label
                        labels = value.get('brushlabels', [])
                        if labels:
                            class_label = labels[0]
                            class_id = self.class_names.index(class_label) if class_label in self.class_names else 1
                            
                            # Add to mask
                            mask = np.maximum(mask, rle_mask * class_id)
            
            return mask
            
        except Exception as e:
            print(f"Error parsing annotation: {e}")
            return None
    
    def _polygon_to_mask(self, polygon_points, height, width):
        """Convert polygon points to binary mask"""
        try:
            from PIL import Image, ImageDraw
            
            # Create a PIL image for drawing
            img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img)
            
            # Draw the polygon
            if len(polygon_points) >= 3:  # Need at least 3 points for a polygon
                draw.polygon(polygon_points, fill=1)
            
            # Convert to numpy array
            mask = np.array(img)
            return mask
            
        except Exception as e:
            print(f"Error converting polygon to mask: {e}")
            return np.zeros((height, width), dtype=np.uint8)
    
    def _rle_to_mask(self, rle_data, height, width):
        """Convert RLE (Run Length Encoding) to binary mask"""
        try:
            # LabelStudio RLE format: [start, length, start, length, ...]
            mask = np.zeros(height * width, dtype=np.uint8)
            
            if len(rle_data) % 2 == 0:
                # Even length - process normally
                for i in range(0, len(rle_data), 2):
                    start = rle_data[i]
                    length = rle_data[i + 1]
                    if start < len(mask) and start + length <= len(mask) and length > 0:
                        mask[start:start + length] = 1
            else:
                # Odd length - truncate to even length
                print(f"⚠️ DEBUG: Odd-length RLE data ({len(rle_data)}), truncating to even length")
                even_length = len(rle_data) - 1
                for i in range(0, even_length, 2):
                    start = rle_data[i]
                    length = rle_data[i + 1]
                    if start < len(mask) and start + length <= len(mask) and length > 0:
                        mask[start:start + length] = 1
            
            return mask.reshape(height, width)
            
        except Exception as e:
            print(f"Error converting RLE to mask: {e}")
            return np.zeros((height, width), dtype=np.uint8)
    
    def __getitem__(self, idx):
        img_blob_name, annotation_blob_name = self.image_annotation_pairs[idx]
        
        # Load image from GCS
        image = self._load_image_from_gcs(img_blob_name)
        if image is None:
            raise ValueError(f"Could not load image: {img_blob_name}")
        
        # Load and parse annotation from GCS
        annotation_data = self._load_annotation_from_gcs(annotation_blob_name)
        mask = self._parse_labelstudio_annotation(annotation_data)
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
            
            # Create binary masks for each class
            for i, class_name in enumerate(self.class_names):
                if i == 0:  # Background class
                    multilabel_mask[i] = (mask == 0).astype(np.float32)
                else:  # Object classes
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


class LabelStudioMinIODatasetNumeric(Dataset):
    """
    MinIO-based dataset for Label Studio annotations with numeric file names
    """
    
    def __init__(self, bucket_name, img_prefix="images/", annotation_prefix="annotations/", 
                 transform=None, multilabel=False, class_names=None):
        """
        Initialize MinIO-based dataset
        
        Args:
            bucket_name: MinIO bucket name
            img_prefix: Prefix for images in the bucket
            annotation_prefix: Prefix for annotations in the bucket
            transform: Albumentations transform
            multilabel: Whether to use multilabel format
            class_names: List of class names (Background first)
        """
        self.bucket_name = bucket_name
        self.img_prefix = img_prefix
        self.annotation_prefix = annotation_prefix
        self.transform = transform
        self.multilabel = multilabel
        self.class_names = class_names or ["Background"]
        
        # MinIO configuration
        import os
        self.endpoint_url = os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
        self.access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
        
        # Initialize MinIO client
        import boto3
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
        
        print(f"✅ Loaded {len(self.image_annotation_pairs)} image-annotation pairs")
    
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
            import cv2
            import numpy as np
            
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
            import json
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=annotation_key)
            annotation_data = response['Body'].read().decode('utf-8')
            return json.loads(annotation_data)
            
        except Exception as e:
            print(f"Error loading annotation from MinIO {annotation_key}: {e}")
            return None
    
    def _parse_labelstudio_annotation(self, annotation_data):
        """Parse Label Studio annotation and create mask"""
        try:
            import cv2
            import numpy as np
            
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
            
            # Create empty mask
            mask = np.zeros((original_height, original_width), dtype=np.uint8)
            
            # Process each annotation result
            for item in result:
                if item.get('type') == 'polygonlabels':
                    value = item.get('value', {})
                    points = value.get('points', [])
                    labels = value.get('polygonlabels', [])
                    
                    if points and labels:
                        # Convert points to polygon format
                        polygon_points = [(int(p[0] * original_width / 100), int(p[1] * original_height / 100)) for p in points]
                        
                        # Create mask for this polygon
                        polygon_mask = self._polygon_to_mask(polygon_points, original_height, original_width)
                        
                        # Get class ID
                        class_label = labels[0]
                        class_id = self.class_names.index(class_label) if class_label in self.class_names else 1
                        
                        # Add to main mask
                        mask = np.maximum(mask, polygon_mask * class_id)
            
            return mask
            
        except Exception as e:
            print(f"Error parsing annotation: {e}")
            return None
    
    def _polygon_to_mask(self, polygon_points, height, width):
        """Convert polygon points to binary mask"""
        try:
            from PIL import Image, ImageDraw
            
            # Create a PIL image for drawing
            img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img)
            
            # Draw the polygon
            if len(polygon_points) >= 3:  # Need at least 3 points for a polygon
                draw.polygon(polygon_points, fill=1)
            
            # Convert to numpy array
            mask = np.array(img)
            return mask
            
        except Exception as e:
            print(f"Error converting polygon to mask: {e}")
            return np.zeros((height, width), dtype=np.uint8)
    
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
        mask = self._parse_labelstudio_annotation(annotation_data)
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
            
            # Create binary masks for each class
            for i, class_name in enumerate(self.class_names):
                if i == 0:  # Background class
                    multilabel_mask[i] = (mask == 0).astype(np.float32)
                else:  # Object classes
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