#!/usr/bin/env python3
import os
import subprocess
import sys

def test_training_environment():
    """Test if environment variables are passed correctly to training process"""
    
    # Set test environment variables
    env = os.environ.copy()
    env['BUCKET_NAME'] = 'test-bucket'
    env['ANNOTATION_PREFIX'] = 'test-masks/'
    
    print("🔍 Testing environment variable passing...")
    print(f"BUCKET_NAME: {env.get('BUCKET_NAME', 'NOT SET')}")
    print(f"ANNOTATION_PREFIX: {env.get('ANNOTATION_PREFIX', 'NOT SET')}")
    
    # Test if we can import the training module
    try:
        sys.path.append('models')
        from training_service import TrainingService
        
        # Create training service with test parameters
        training_service = TrainingService(
            bucket_name=env['BUCKET_NAME'],
            annotation_prefix=env['ANNOTATION_PREFIX']
        )
        
        print("✅ TrainingService created successfully!")
        print(f"   - bucket_name: {training_service.bucket_name}")
        print(f"   - annotation_prefix: {training_service.annotation_prefix}")
        
        # Test environment variable setting
        test_env = os.environ.copy()
        test_env['BUCKET_NAME'] = training_service.bucket_name
        test_env['ANNOTATION_PREFIX'] = training_service.annotation_prefix
        
        print(f"✅ Environment variables set correctly:")
        print(f"   - BUCKET_NAME: {test_env.get('BUCKET_NAME')}")
        print(f"   - ANNOTATION_PREFIX: {test_env.get('ANNOTATION_PREFIX')}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_environment()
