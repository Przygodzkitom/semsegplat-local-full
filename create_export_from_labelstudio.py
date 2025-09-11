#!/usr/bin/env python3
"""
Script to create a Label Studio export file by reading directly from Label Studio database
This bypasses the MinIO export storage issue
"""

import os
import json
import sqlite3
import requests
from datetime import datetime

def create_export_from_labelstudio_db():
    """Create export file by reading from Label Studio database"""
    
    # Path to Label Studio database
    db_path = "label-studio-data/label_studio.sqlite3"
    
    if not os.path.exists(db_path):
        print(f"❌ Label Studio database not found: {db_path}")
        return False
    
    try:
        # Connect to Label Studio database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query to get tasks with annotations
        query = """
        SELECT 
            t.id as task_id,
            t.data as task_data,
            a.id as annotation_id,
            a.result as annotation_result,
            a.created_at as annotation_created_at
        FROM task_task t
        LEFT JOIN task_annotation a ON t.id = a.task_id
        WHERE a.result IS NOT NULL
        ORDER BY t.id, a.id
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            print("❌ No annotations found in Label Studio database")
            conn.close()
            return False
        
        print(f"✅ Found {len(rows)} annotation records in database")
        
        # Group by task_id
        tasks = {}
        for row in rows:
            task_id, task_data, annotation_id, annotation_result, annotation_created_at = row
            
            if task_id not in tasks:
                tasks[task_id] = {
                    'id': task_id,
                    'data': json.loads(task_data) if task_data else {},
                    'annotations': []
                }
            
            if annotation_result:
                tasks[task_id]['annotations'].append({
                    'id': annotation_id,
                    'result': json.loads(annotation_result),
                    'created_at': annotation_created_at
                })
        
        # Convert to Label Studio export format
        export_data = []
        for task in tasks.values():
            if task['annotations']:  # Only include tasks with annotations
                export_data.append(task)
        
        if not export_data:
            print("❌ No valid annotations found")
            conn.close()
            return False
        
        # Create export directory
        export_dir = "label-studio-data/export/"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        filename = f"project-1-at-{timestamp}-db-export.json"
        filepath = os.path.join(export_dir, filename)
        
        # Save export file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Export file created: {filepath}")
        print(f"📊 Total tasks with annotations: {len(export_data)}")
        
        # Show some details
        total_annotations = sum(len(task['annotations']) for task in export_data)
        print(f"📊 Total annotations: {total_annotations}")
        
        conn.close()
        return filepath
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        if 'conn' in locals():
            conn.close()
        return False

def create_export_via_api():
    """Alternative: Create export via Label Studio API"""
    
    try:
        # Label Studio API endpoint
        base_url = "http://localhost:8080"
        project_id = 1  # Assuming project 1
        
        # Try to get project info first
        response = requests.get(f"{base_url}/api/projects/{project_id}/")
        if response.status_code != 200:
            print(f"❌ Cannot connect to Label Studio API: {response.status_code}")
            return False
        
        project_data = response.json()
        print(f"✅ Connected to Label Studio project: {project_data.get('title', 'Unknown')}")
        
        # Get tasks with annotations
        tasks_response = requests.get(f"{base_url}/api/projects/{project_id}/tasks/")
        if tasks_response.status_code != 200:
            print(f"❌ Cannot get tasks: {tasks_response.status_code}")
            return False
        
        tasks_data = tasks_response.json()
        tasks = tasks_data.get('results', [])
        
        if not tasks:
            print("❌ No tasks found in project")
            return False
        
        # Filter tasks with annotations
        annotated_tasks = [task for task in tasks if task.get('annotations')]
        
        if not annotated_tasks:
            print("❌ No annotated tasks found")
            return False
        
        print(f"✅ Found {len(annotated_tasks)} annotated tasks via API")
        
        # Create export directory
        export_dir = "label-studio-data/export/"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        filename = f"project-1-at-{timestamp}-api-export.json"
        filepath = os.path.join(export_dir, filename)
        
        # Save export file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(annotated_tasks, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Export file created: {filepath}")
        print(f"📊 Total annotated tasks: {len(annotated_tasks)}")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Error with API export: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Creating Label Studio export file...")
    print("Trying database method first...")
    
    result = create_export_from_labelstudio_db()
    
    if not result:
        print("\nTrying API method...")
        result = create_export_via_api()
    
    if result:
        print(f"\n✅ Success! Export file created: {result}")
        print("You can now run batch evaluation in Streamlit.")
    else:
        print("\n❌ Failed to create export file.")
        print("Please check if Label Studio is running and has annotations.")
