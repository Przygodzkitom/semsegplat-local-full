#!/usr/bin/env python3
"""
Training Service - Runs training as a separate process with proper resource isolation
This service can be called from Streamlit without causing resource contention
"""

import os
import sys
import json
import signal
import subprocess
import threading
import time
import psutil
from pathlib import Path

class TrainingService:
    def __init__(self, bucket_name="segmentation-platform", annotation_prefix="masks/"):
        self.training_process = None
        # Use absolute paths to ensure files are created in the correct location
        self.progress_file = "/app/training_progress.json"
        self.log_file = "/app/training.log"
        self.pid_file = "/app/training.pid"
        self.is_running = False
        self.bucket_name = bucket_name
        self.annotation_prefix = annotation_prefix
        self.num_epochs = 100
        
    # Annotation type is configured directly in Label Studio
    # No need to read from config file
        
    def start_training(self, num_epochs=100):
        """Start training in a separate process with proper isolation"""
        print(f"🔍 DEBUG: start_training() called with bucket_name='{self.bucket_name}', annotation_prefix='{self.annotation_prefix}'")
        
        if self.is_running:
            print("🔍 DEBUG: Training already running, returning False")
            return False, "Training already running"
            
        try:
            print("🔍 DEBUG: Clearing previous progress...")
            # Clear previous progress
            self._clear_progress()
            
            print("🔍 DEBUG: Setting environment variables...")
            # Set environment variables for proper isolation
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # GPU detection will be handled by the training script
            # Don't force CUDA_VISIBLE_DEVICES here, let the script detect
            
            # Set environment variables for bucket and prefix
            env['BUCKET_NAME'] = self.bucket_name
            env['ANNOTATION_PREFIX'] = self.annotation_prefix
            env['NUM_EPOCHS'] = str(num_epochs)
            self.num_epochs = num_epochs

            # Create test split before training (reserves images never seen by model)
            test_split_file = self._create_test_split_if_needed()
            if test_split_file:
                env['TEST_SPLIT_FILE'] = test_split_file

            # Detect annotation type and choose appropriate training script
            training_script = self._detect_and_choose_training_script()
            print(f"🔍 DEBUG: Using training script: {training_script}")
            
            print(f"🔍 DEBUG: Environment variables set - BUCKET_NAME='{env.get('BUCKET_NAME')}', ANNOTATION_PREFIX='{env.get('ANNOTATION_PREFIX')}'")
            print(f"🔍 DEBUG: Starting subprocess with command: {sys.executable} {training_script}")
            
            # Start training process with proper isolation
            self.training_process = subprocess.Popen(
                [sys.executable, training_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=0,
                env=env,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Wait a moment to see if the process starts successfully
            import time
            time.sleep(2)
            
            # Check if process is still running
            if self.training_process.poll() is not None:
                # Process died immediately
                return_code = self.training_process.returncode
                print(f"🔍 DEBUG: Training process died immediately with return code: {return_code}")
                # Try to read any error output
                try:
                    error_output = self.training_process.stdout.read()
                    print(f"🔍 DEBUG: Error output: {error_output}")
                except:
                    pass
                return False, f"Training process failed to start (return code: {return_code})"
            
            print(f"🔍 DEBUG: Subprocess started with PID: {self.training_process.pid}")
            # Persist PID so stop_training() works across Streamlit rerenders
            with open(self.pid_file, 'w') as f:
                f.write(str(self.training_process.pid))
            self.is_running = True
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_training)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            print("🔍 DEBUG: Training started successfully")
            return True, "Training started successfully"
            
        except Exception as e:
            print(f"🔍 DEBUG: Exception in start_training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, f"Failed to start training: {str(e)}"
    
    def stop_training(self):
        """Stop the training process gracefully"""
        # Resolve the process group ID — either from the live process handle
        # or from the PID file (needed when called from a fresh Streamlit rerender
        # that has no direct reference to the original subprocess).
        pgid = None

        if self.training_process:
            try:
                pgid = os.getpgid(self.training_process.pid)
            except (ProcessLookupError, OSError):
                pass

        if pgid is None and os.path.exists(self.pid_file):
            try:
                with open(self.pid_file) as f:
                    pid = int(f.read().strip())
                pgid = os.getpgid(pid)
            except (ValueError, ProcessLookupError, OSError):
                pass

        if pgid is None:
            return False, "No training process running"

        try:
            # Send SIGINT so Python's KeyboardInterrupt handler runs and saves
            # the partial checkpoint before exiting.
            os.killpg(pgid, signal.SIGINT)

            # Wait up to 60 s for the script to finish saving
            try:
                if self.training_process:
                    self.training_process.wait(timeout=60)
                else:
                    time.sleep(10)  # give the process group time to save and exit
            except subprocess.TimeoutExpired:
                # Graceful save timed out — force kill
                os.killpg(pgid, signal.SIGKILL)
                if self.training_process:
                    self.training_process.wait()

            self.is_running = False
            self.training_process = None

            # Remove PID file
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)

            return True, "Training stopped successfully"

        except Exception as e:
            return False, f"Failed to stop training: {str(e)}"
    
    def detect_running_training(self):
        """Detect if there's a training process running and reconnect to it"""
        # First check: if the progress file says the run is done, don't reconnect
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)
                if data.get('status') in ('completed', 'failed', 'interrupted'):
                    return False
            except Exception:
                pass

        try:
            # Use a specific pattern so we only match the actual training scripts
            result = subprocess.run(
                ['pgrep', '-f', r'training_(brush|polygon)\.py'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                self.is_running = True
                return True
        except Exception:
            pass
        return False
    
    def get_status(self):
        """Get current training status"""
        print(f"🔍 DEBUG: get_status() called, is_running={self.is_running}")
        
        # First, check if there's actually a training process running
        if self.training_process:
            print(f"🔍 DEBUG: We have a tracked training process")
            # Check if our tracked process is still alive
            if self.training_process.poll() is not None:
                print(f"🔍 DEBUG: Training process has ended")
                self.is_running = False
                self.training_process = None  # clear so we fall through to progress file
            else:
                print(f"🔍 DEBUG: Training process is still running")
        
        # Check if there's a training process running by checking the progress file
        if os.path.exists(self.progress_file):
            try:
                print(f"🔍 DEBUG: Progress file exists, reading it")
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                print(f"🔍 DEBUG: Progress data: {progress_data}")
                
                status_val = progress_data.get('status')
                if status_val in ('running', 'initializing', 'interrupted'):
                    print(f"🔍 DEBUG: Progress file shows status={status_val}")
                    self.is_running = True
                    return progress_data
                elif status_val in ('completed', 'failed'):
                    print(f"🔍 DEBUG: Progress file shows completed/failed status")
                    self.is_running = False
                    return progress_data
            except Exception as e:
                print(f"🔍 DEBUG: Error reading progress file: {e}")
                import traceback
                traceback.print_exc()
        
        # Default idle state if no progress file or no running status
        print(f"🔍 DEBUG: Returning idle state")
        return {
            'status': 'idle',
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': self.num_epochs,
            'log': []
        }
    
    def _monitor_training(self):
        """Monitor training process and update progress file"""
        print("🔍 DEBUG: Monitoring thread started")
        while self.is_running and self.training_process:
            try:
                # Read output line by line
                output = self.training_process.stdout.readline()
                if output == '' and self.training_process.poll() is not None:
                    print("🔍 DEBUG: Training process ended (no more output)")
                    break
                
                if output:
                    print(f"🔍 Training output: {output.strip()}")
                    # Parse epoch information
                    epoch_info = self._parse_epoch_info(output.strip())
                    if epoch_info:
                        self._update_progress(epoch_info)
                    
                    # Append to log
                    self._append_to_log(output.strip())
                    
            except Exception as e:
                print(f"🔍 Error monitoring training: {e}")
                break
        
        # Final status update
        if self.training_process:
            return_code = self.training_process.poll()
            final_status = 'completed' if return_code == 0 else 'failed'
            print(f"🔍 DEBUG: Training process final status: {final_status} (return code: {return_code})")
            self._update_progress({'status': final_status})
        
        self.is_running = False
        print("🔍 DEBUG: Monitoring thread ended")
    
    def _parse_epoch_info(self, line):
        """Parse epoch information from training output"""
        if "Epoch " in line and "/" in line:
            try:
                # Extract epoch number and total epochs
                epoch_part = line.split("Epoch ")[1]
                if "/" in epoch_part:
                    current_epoch_str, total_epochs_str = epoch_part.split("/")
                    current_epoch = int(current_epoch_str)
                    total_epochs = int(total_epochs_str)
                    return {
                        'current_epoch': current_epoch,
                        'total_epochs': total_epochs,
                        'progress': (current_epoch / total_epochs) * 100
                    }
            except:
                pass
        return None
    
    def _update_progress(self, data):
        """Update progress file"""
        try:
            print(f"🔍 DEBUG: TrainingService._update_progress called with data: {data}")
            current_data = {}
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    current_data = json.load(f)
                print(f"🔍 DEBUG: Current progress data: {current_data}")
            
            current_data.update(data)
            print(f"🔍 DEBUG: Updated progress data: {current_data}")
            
            with open(self.progress_file, 'w') as f:
                json.dump(current_data, f, indent=2)
            print(f"🔍 DEBUG: Progress file updated by TrainingService")
                
        except Exception as e:
            print(f"🔍 ERROR: Error updating progress in TrainingService: {e}")
            import traceback
            traceback.print_exc()
    
    def _append_to_log(self, line):
        """Append line to log file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{line}\n")
        except Exception as e:
            print(f"Error appending to log: {e}")
    
    def _create_test_split_if_needed(self):
        """Reserve 20% of annotated images as a held-out test set before training.

        If the file already exists the same split is reused across training runs,
        so test images are never seen by the model.
        """
        test_split_file = "/app/models/checkpoints/test_split.json"

        if os.path.exists(test_split_file):
            try:
                with open(test_split_file) as f:
                    existing = json.load(f)
                n_test = len(existing.get('test_image_keys', []))
                print(f"✅ Existing test split loaded: {n_test} images reserved for evaluation")
                return test_split_file
            except Exception as e:
                print(f"⚠️ Could not read existing test split ({e}), recreating")

        print("🔍 Creating test split from annotated images...")
        try:
            import boto3
            import random

            endpoint_url = os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
            access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
            secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')

            s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name='us-east-1'
            )

            response = s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.annotation_prefix
            )

            if 'Contents' not in response:
                print("⚠️ No annotations found, skipping test split creation")
                return None

            image_keys = []
            for obj in response['Contents']:
                if obj['Key'].endswith('/') or obj['Size'] == 0:
                    continue
                if '.' not in obj['Key'].split('/')[-1]:
                    try:
                        ann_response = s3_client.get_object(
                            Bucket=self.bucket_name, Key=obj['Key']
                        )
                        annotation_data = json.loads(ann_response['Body'].read().decode('utf-8'))
                        if annotation_data and 'task' in annotation_data:
                            image_path = annotation_data['task'].get('data', {}).get('image', '')
                            if image_path.startswith('s3://'):
                                image_key = image_path.replace(f's3://{self.bucket_name}/', '')
                                image_keys.append(image_key)
                    except Exception as e:
                        print(f"⚠️ Error reading annotation {obj['Key']}: {e}")
                        continue

            if not image_keys:
                print("⚠️ No image keys found, skipping test split creation")
                return None

            n_test = max(1, int(len(image_keys) * 0.2))
            random.shuffle(image_keys)
            test_keys = image_keys[:n_test]

            from datetime import datetime
            os.makedirs("/app/models/checkpoints", exist_ok=True)
            split_data = {
                'created_at': datetime.now().isoformat(),
                'test_image_keys': test_keys,
                'total_annotated': len(image_keys),
                'test_fraction': round(n_test / len(image_keys), 3)
            }
            with open(test_split_file, 'w') as f:
                json.dump(split_data, f, indent=2)

            print(f"✅ Test split created: {n_test}/{len(image_keys)} images reserved for evaluation")
            return test_split_file

        except Exception as e:
            print(f"⚠️ Could not create test split: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _detect_and_choose_training_script(self):
        """Detect annotation type and choose appropriate training script"""
        try:
            # Import the annotation type detector
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
            from annotation_type_detector import AnnotationTypeDetector
            
            # Detect annotation type
            detector = AnnotationTypeDetector(self.bucket_name, self.annotation_prefix)
            detection = detector.detect_annotation_type()
            
            print(f"🔍 DEBUG: Annotation type detection result: {detection}")
            
            # Choose training script based on detection
            if detection['type'] == 'polygon':
                training_script = "/app/models/training_polygon.py"
                print(f"🎯 POLYGON annotations detected - using polygon training script")
            elif detection['type'] == 'brush':
                training_script = "/app/models/training_brush.py"
                print(f"🎨 BRUSH annotations detected - using brush training script")
            else:
                # Default to brush training script for mixed/unknown types
                training_script = "/app/models/training_brush.py"
                print(f"⚠️ Mixed/unknown annotation types detected - defaulting to brush training script")
            
            return training_script
            
        except Exception as e:
            raise RuntimeError(f"Failed to detect annotation type: {e}. Cannot proceed with training.")
    
    def _clear_progress(self):
        """Clear previous progress files and memory state"""
        try:
            # Clear progress files
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
            
            # Clear memory state
            self._clear_memory_state()
            
        except Exception as e:
            print(f"Error clearing progress: {e}")
    
    def _clear_memory_state(self):
        """Clear memory state between training sessions"""
        try:
            import gc
            import torch
            
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear any remaining Python objects
            gc.collect()
            
            print("🧹 Memory state cleared between training sessions")
            
        except Exception as e:
            print(f"Warning: Could not clear memory state: {e}")

# Global training service instance
training_service = TrainingService()

def main():
    """Main function for running training service"""
    if len(sys.argv) < 2:
        print("Usage: python training_service.py [start|stop|status]")
        return
    
    command = sys.argv[1]
    
    if command == "start":
        success, message = training_service.start_training()
        print(json.dumps({"success": success, "message": message}))
        
    elif command == "stop":
        success, message = training_service.stop_training()
        print(json.dumps({"success": success, "message": message}))
        
    elif command == "status":
        status = training_service.get_status()
        print(json.dumps(status))
        
    else:
        print("Unknown command. Use: start, stop, or status")

if __name__ == "__main__":
    main() 