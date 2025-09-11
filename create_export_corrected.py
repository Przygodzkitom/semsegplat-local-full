#!/usr/bin/env python3
"""
Script to create a Label Studio export file by reading from the correct database tables
"""

import os
import json
import sqlite3
from datetime import datetime

def create_export_from_database():
    """Create export file by reading from Label Studio database with correct table names"""
    
    # Path to Label Studio database
    db_path = "label-studio-data/label_studio.sqlite3"
    
    if not os.path.exists(db_path):
        print(f"❌ Label Studio database not found: {db_path}")
        return False
    
    try:
        # Connect to Label Studio database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query to get tasks with annotations using correct table names
        query = """
        SELECT 
            t.id as task_id,
            t.data as task_data,
            tc.id as completion_id,
            tc.result as completion_result,
            tc.created_at as completion_created_at
        FROM task t
        LEFT JOIN task_completion tc ON t.id = tc.task_id
        WHERE tc.result IS NOT NULL
        ORDER BY t.id, tc.id
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
            task_id, task_data, completion_id, completion_result, completion_created_at = row
            
            if task_id not in tasks:
                tasks[task_id] = {
                    'id': task_id,
                    'data': json.loads(task_data) if task_data else {},
                    'annotations': []
                }
            
            if completion_result:
                tasks[task_id]['annotations'].append({
                    'id': completion_id,
                    'result': json.loads(completion_result),
                    'created_at': completion_created_at
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
        
        # Show class information if available
        classes = set()
        for task in export_data:
            for annotation in task['annotations']:
                result = annotation.get('result', [])
                for item in result:
                    if 'value' in item and 'brushlabels' in item['value']:
                        classes.update(item['value']['brushlabels'])
                    elif 'value' in item and 'polygonlabels' in item['value']:
                        classes.update(item['value']['polygonlabels'])
        
        if classes:
            print(f"🎨 Classes found: {list(classes)}")
        
        conn.close()
        return filepath
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        if 'conn' in locals():
            conn.close()
        return False

if __name__ == "__main__":
    print("🔄 Creating Label Studio export file from database...")
    result = create_export_from_database()
    
    if result:
        print(f"\n✅ Success! Export file created: {result}")
        print("You can now run batch evaluation in Streamlit.")
    else:
        print("\n❌ Failed to create export file.")
        print("Please check if Label Studio has annotations.")
