#!/usr/bin/env python3
"""
Script to trigger Label Studio export and create JSON files for batch inference
"""

import os
import json
import requests
import time
from datetime import datetime

def trigger_labelstudio_export():
    """Trigger Label Studio export via API"""
    
    base_url = "http://localhost:8080"
    project_id = 1  # Assuming project 1
    
    try:
        # First, check if project exists and has annotations
        print("🔍 Checking Label Studio project...")
        project_response = requests.get(f"{base_url}/api/projects/{project_id}/")
        
        if project_response.status_code != 200:
            print(f"❌ Cannot connect to Label Studio: {project_response.status_code}")
            return False
        
        project_data = project_response.json()
        print(f"✅ Connected to project: {project_data.get('title', 'Unknown')}")
        
        # Get tasks to check for annotations
        tasks_response = requests.get(f"{base_url}/api/projects/{project_id}/tasks/")
        if tasks_response.status_code != 200:
            print(f"❌ Cannot get tasks: {tasks_response.status_code}")
            return False
        
        tasks_data = tasks_response.json()
        tasks = tasks_data.get('results', [])
        
        if not tasks:
            print("❌ No tasks found in project")
            return False
        
        # Count annotated tasks
        annotated_tasks = [task for task in tasks if task.get('annotations')]
        print(f"📊 Found {len(tasks)} total tasks, {len(annotated_tasks)} with annotations")
        
        if not annotated_tasks:
            print("❌ No annotated tasks found. Please complete some annotations first.")
            return False
        
        # Create export directory
        export_dir = "label-studio-data/export/"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        filename = f"project-{project_id}-at-{timestamp}-export.json"
        filepath = os.path.join(export_dir, filename)
        
        # Save the annotated tasks as export file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(annotated_tasks, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Export file created: {filepath}")
        print(f"📊 Exported {len(annotated_tasks)} annotated tasks")
        
        # Show some details about the annotations
        total_annotations = sum(len(task.get('annotations', [])) for task in annotated_tasks)
        print(f"📊 Total annotations: {total_annotations}")
        
        # Show class information if available
        classes = set()
        for task in annotated_tasks:
            for annotation in task.get('annotations', []):
                result = annotation.get('result', [])
                for item in result:
                    if 'value' in item and 'brushlabels' in item['value']:
                        classes.update(item['value']['brushlabels'])
                    elif 'value' in item and 'polygonlabels' in item['value']:
                        classes.update(item['value']['polygonlabels'])
        
        if classes:
            print(f"🎨 Classes found: {list(classes)}")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Triggering Label Studio export...")
    result = trigger_labelstudio_export()
    
    if result:
        print(f"\n✅ Success! Export file created: {result}")
        print("You can now run batch evaluation in Streamlit.")
    else:
        print("\n❌ Failed to create export file.")
        print("Please check if Label Studio is running and has annotations.")
