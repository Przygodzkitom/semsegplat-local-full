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
        self.is_running = False
        self.bucket_name = bucket_name
        self.annotation_prefix = annotation_prefix
        
    # Annotation type is configured directly in Label Studio
    # No need to read from config file
        
    def start_training(self):
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
        if not self.is_running or not self.training_process:
            return False, "No training process running"
            
        try:
            # Send SIGTERM to the process group
            os.killpg(os.getpgid(self.training_process.pid), signal.SIGTERM)
            
            # Wait for graceful shutdown
            try:
                self.training_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                os.killpg(os.getpgid(self.training_process.pid), signal.SIGKILL)
                self.training_process.wait()
            
            self.is_running = False
            self.training_process = None
            
            return True, "Training stopped successfully"
            
        except Exception as e:
            return False, f"Failed to stop training: {str(e)}"
    
    def detect_running_training(self):
        """Detect if there's a training process running and reconnect to it"""
        try:
            import subprocess
            # Check for any training process (both polygon and brush)
            result = subprocess.run(['pgrep', '-f', 'training'], capture_output=True, text=True)
            if result.returncode == 0:
                # There's a training process running
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
                return {
                    'status': 'completed' if self.training_process.returncode == 0 else 'failed',
                    'progress': 0,
                    'current_epoch': 0,
                    'total_epochs': 100,
                    'log': []
                }
            else:
                print(f"🔍 DEBUG: Training process is still running")
        
        # Check if there's a training process running by checking the progress file
        if os.path.exists(self.progress_file):
            try:
                print(f"🔍 DEBUG: Progress file exists, reading it")
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                print(f"🔍 DEBUG: Progress data: {progress_data}")
                
                # If the progress file shows 'running' status, we assume training is running
                if progress_data.get('status') == 'running':
                    print(f"🔍 DEBUG: Progress file shows running status, setting is_running=True")
                    self.is_running = True
                    return progress_data
                elif progress_data.get('status') in ['completed', 'failed']:
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
            'total_epochs': 100,
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
                training_script = "/app/models/training_brush_minimal.py"
                print(f"🎨 BRUSH annotations detected - using minimal brush training script")
            else:
                # Default to brush training script for mixed/unknown types
                training_script = "/app/models/training_brush.py"
                print(f"⚠️ Mixed/unknown annotation types detected - defaulting to brush training script")
            
            return training_script
            
        except Exception as e:
            print(f"⚠️ Error detecting annotation type: {e}")
            print(f"🔄 Falling back to original training script")
            return "/app/models/training.py"
    
    def _clear_progress(self):
        """Clear previous progress files and memory state"""
        try:
            # Clear progress files
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            
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