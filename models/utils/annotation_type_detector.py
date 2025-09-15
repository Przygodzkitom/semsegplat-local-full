#!/usr/bin/env python3
"""
Annotation Type Detection Utility
Detects whether Label Studio annotations use polygon or brush format
and determines how background class is handled
"""

import json
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Dict, List, Set, Tuple, Optional

class AnnotationTypeDetector:
    def __init__(self, bucket_name: str, annotation_prefix: str = "masks/"):
        """
        Initialize annotation type detector
        
        Args:
            bucket_name: MinIO bucket name
            annotation_prefix: Prefix for annotations in the bucket
        """
        self.bucket_name = bucket_name
        self.annotation_prefix = annotation_prefix
        
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
            region_name='us-east-1'  # MinIO default region
        )
    
    def detect_annotation_type(self) -> Dict:
        """
        Detect annotation type and background handling from Label Studio annotations
        
        Returns:
            Dictionary with annotation type information:
            {
                'type': 'polygon' | 'brush' | 'mixed',
                'has_explicit_background': bool,
                'background_handling': 'automatic' | 'explicit' | 'none',
                'class_names': List[str],
                'sample_annotations': int
            }
        """
        try:
            print(f"🔍 DEBUG: AnnotationTypeDetector.detect_annotation_type() called")
            print(f"🔍 DEBUG: Looking for annotations in bucket '{self.bucket_name}' with prefix '{self.annotation_prefix}'")
            
            # Get all annotation files from MinIO
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.annotation_prefix
            )
            
            if 'Contents' not in response:
                print(f"❌ No objects found in bucket {self.bucket_name} with prefix {self.annotation_prefix}")
                return {
                    'type': 'unknown',
                    'has_explicit_background': False,
                    'background_handling': 'none',
                    'class_names': [],
                    'sample_annotations': 0
                }
            
            print(f"✅ Found {len(response['Contents'])} objects in bucket with prefix {self.annotation_prefix}")
            
            annotation_types = set()
            has_explicit_background = False
            all_class_names = set()
            sample_count = 0
            
            # Sample up to 10 annotations for analysis
            max_samples = min(10, len(response['Contents']))
            
            for i, obj in enumerate(response['Contents']):
                if i >= max_samples:
                    break
                    
                # Skip directory placeholders and only accept actual files
                if obj['Key'].endswith('/') or obj['Size'] == 0:
                    continue
                
                # Accept JSON annotation files (with or without extension)
                if obj['Key'].endswith('.json') or not '.' in obj['Key'].split('/')[-1]:
                    try:
                        # Load annotation from MinIO
                        annotation_data = self._load_annotation_from_minio(obj['Key'])
                        if annotation_data:
                            # Analyze annotation type and classes
                            analysis = self._analyze_annotation(annotation_data)
                            
                            if analysis['type']:
                                annotation_types.add(analysis['type'])
                            
                            if analysis['has_background']:
                                has_explicit_background = True
                            
                            all_class_names.update(analysis['class_names'])
                            sample_count += 1
                            
                    except Exception as e:
                        print(f"Error processing annotation {obj['Key']}: {e}")
                        continue
            
            # Determine overall annotation type
            if len(annotation_types) == 1:
                annotation_type = list(annotation_types)[0]
            elif len(annotation_types) > 1:
                annotation_type = 'mixed'
            else:
                annotation_type = 'unknown'
            
            # Determine background handling
            if annotation_type == 'polygon':
                background_handling = 'automatic'  # Polygon always has automatic background
            elif annotation_type == 'brush':
                if has_explicit_background:
                    background_handling = 'explicit'  # User chose to include background
                else:
                    background_handling = 'none'  # User chose not to include background
            else:
                background_handling = 'mixed'
            
            result = {
                'type': annotation_type,
                'has_explicit_background': has_explicit_background,
                'background_handling': background_handling,
                'class_names': sorted(list(all_class_names)),
                'sample_annotations': sample_count
            }
            
            print(f"🔍 DEBUG: Annotation type detection result: {result}")
            return result
            
        except Exception as e:
            print(f"Error detecting annotation type: {e}")
            return {
                'type': 'unknown',
                'has_explicit_background': False,
                'background_handling': 'none',
                'class_names': [],
                'sample_annotations': 0
            }
    
    def _load_annotation_from_minio(self, object_key: str) -> Dict:
        """Load annotation from MinIO"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_key)
            annotation_data = response['Body'].read().decode('utf-8')
            return json.loads(annotation_data)
        except Exception as e:
            print(f"Error loading annotation from MinIO {object_key}: {e}")
            return None
    
    def _analyze_annotation(self, annotation_data: Dict) -> Dict:
        """Analyze a single annotation to determine type and classes"""
        analysis = {
            'type': None,
            'has_background': False,
            'class_names': set()
        }
        
        try:
            result = annotation_data.get('result', [])
            
            for item in result:
                item_type = item.get('type')
                
                if item_type == 'polygonlabels':
                    analysis['type'] = 'polygon'
                    value = item.get('value', {})
                    labels = value.get('polygonlabels', [])
                    analysis['class_names'].update(labels)
                    
                    # Check if background is explicitly labeled
                    if 'Background' in labels or 'background' in labels:
                        analysis['has_background'] = True
                
                elif item_type == 'brushlabels':
                    analysis['type'] = 'brush'
                    value = item.get('value', {})
                    labels = value.get('brushlabels', [])
                    analysis['class_names'].update(labels)
                    
                    # Check if background is explicitly labeled
                    if 'Background' in labels or 'background' in labels:
                        analysis['has_background'] = True
                
                elif item_type == 'labels':
                    # General labels (could be any type)
                    value = item.get('value', {})
                    labels = value.get('labels', [])
                    analysis['class_names'].update(labels)
                    
                    if 'Background' in labels or 'background' in labels:
                        analysis['has_background'] = True
            
            # Convert set to list for JSON serialization
            analysis['class_names'] = list(analysis['class_names'])
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing annotation: {e}")
            return analysis
    
    def get_recommended_class_config(self) -> Dict:
        """
        Get recommended class configuration based on detected annotation type
        
        Returns:
            Dictionary with recommended class configuration
        """
        detection = self.detect_annotation_type()
        
        if detection['type'] == 'polygon':
            # For polygon annotations, always include Background as first class
            class_names = ['Background'] + [name for name in detection['class_names'] if name.lower() != 'background']
            return {
                'class_names': class_names,
                'annotation_type': 'polygon',
                'background_handling': 'automatic',
                'recommendation': 'Use polygon training script - background is automatically handled'
            }
        
        elif detection['type'] == 'brush':
            if detection['has_explicit_background']:
                # User explicitly included background in brush annotations
                class_names = detection['class_names']
                return {
                    'class_names': class_names,
                    'annotation_type': 'brush',
                    'background_handling': 'explicit',
                    'recommendation': 'Use brush training script - background is explicitly defined'
                }
            else:
                # User did not include background in brush annotations
                class_names = detection['class_names']
                return {
                    'class_names': class_names,
                    'annotation_type': 'brush',
                    'background_handling': 'none',
                    'recommendation': 'Use brush training script - no background class needed'
                }
        
        else:
            # Mixed or unknown - default to brush handling
            class_names = detection['class_names'] if detection['class_names'] else ['Object']
            return {
                'class_names': class_names,
                'annotation_type': 'mixed',
                'background_handling': 'mixed',
                'recommendation': 'Mixed annotation types detected - using brush training script as default'
            }

