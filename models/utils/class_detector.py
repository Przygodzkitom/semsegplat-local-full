#!/usr/bin/env python3
"""
Class Detection Utility
Automatically detects classes from Label Studio annotations
"""

import json
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List, Dict, Set

class ClassDetector:
    def __init__(self, bucket_name: str, annotation_prefix: str = "annotations/"):
        """
        Initialize class detector
        
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
    
    def detect_classes(self) -> Dict[str, int]:
        """
        Detect all unique classes from Label Studio annotations
        
        Returns:
            Dictionary mapping class names to their counts
        """
        class_counts = {}
        
        try:
            # Get all annotation files from MinIO
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.annotation_prefix
            )
            
            if 'Contents' not in response:
                print(f"No objects found in bucket {self.bucket_name} with prefix {self.annotation_prefix}")
                return {}
            
            for obj in response['Contents']:
                # Skip directory placeholders and only accept actual files
                if obj['Key'].endswith('/') or obj['Size'] == 0:
                    continue
                
                # Accept JSON annotation files (with or without extension)
                if obj['Key'].endswith('.json') or not '.' in obj['Key'].split('/')[-1]:
                    try:
                        # Load annotation from MinIO
                        annotation_data = self._load_annotation_from_minio(obj['Key'])
                        if annotation_data:
                            # Extract classes from annotation
                            classes = self._extract_classes_from_annotation(annotation_data)
                            for class_name in classes:
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    except Exception as e:
                        print(f"Error processing annotation {obj['Key']}: {e}")
                        continue
            
            return class_counts
            
        except Exception as e:
            print(f"Error detecting classes: {e}")
            return {}
    
    def _load_annotation_from_minio(self, object_key: str) -> Dict:
        """Load annotation from MinIO"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_key)
            annotation_data = response['Body'].read().decode('utf-8')
            return json.loads(annotation_data)
        except Exception as e:
            print(f"Error loading annotation from MinIO {object_key}: {e}")
            return None
    
    def _extract_classes_from_annotation(self, annotation_data: Dict) -> Set[str]:
        """Extract class names from Label Studio annotation"""
        classes = set()
        
        try:
            # Handle different annotation formats
            if isinstance(annotation_data, list):
                # List of annotations
                for annotation in annotation_data:
                    classes.update(self._extract_classes_from_single_annotation(annotation))
            elif isinstance(annotation_data, dict):
                # Single annotation
                classes.update(self._extract_classes_from_single_annotation(annotation_data))
            
            return classes
            
        except Exception as e:
            print(f"Error extracting classes from annotation: {e}")
            return set()
    
    def _extract_classes_from_single_annotation(self, annotation: Dict) -> Set[str]:
        """Extract class names from a single Label Studio annotation"""
        classes = set()
        
        try:
            result = annotation.get('result', [])
            
            for item in result:
                if item.get('type') == 'polygonlabels':
                    value = item.get('value', {})
                    labels = value.get('polygonlabels', [])
                    classes.update(labels)
                
                elif item.get('type') == 'brushlabels':
                    value = item.get('value', {})
                    labels = value.get('brushlabels', [])
                    classes.update(labels)
                
                elif item.get('type') == 'labels':
                    value = item.get('value', {})
                    labels = value.get('labels', [])
                    classes.update(labels)
            
            return classes
            
        except Exception as e:
            print(f"Error extracting classes from single annotation: {e}")
            return set()
    
    def get_class_statistics(self) -> Dict:
        """
        Get detailed statistics about detected classes
        
        Returns:
            Dictionary with class statistics
        """
        class_counts = self.detect_classes()
        
        if not class_counts:
            return {
                'total_annotations': 0,
                'classes': {},
                'class_names': [],
                'num_classes': 0
            }
        
        # Sort classes by count (descending)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_annotations': sum(class_counts.values()),
            'classes': dict(sorted_classes),
            'class_names': [name for name, _ in sorted_classes],
            'num_classes': len(class_counts)
        } 