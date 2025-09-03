import os
import io
from PIL import Image
import streamlit as st
from typing import Tuple, Optional, List
import mimetypes

def detect_image_format(file_data) -> Tuple[str, str]:
    """
    Detect the format of an uploaded image file
    
    Args:
        file_data: StreamlitUploadedFile object
        
    Returns:
        Tuple of (detected_format, mime_type)
    """
    try:
        # Reset file pointer
        file_data.seek(0)
        
        # Try to detect format using PIL
        with Image.open(file_data) as img:
            detected_format = img.format.upper() if img.format else "UNKNOWN"
        
        # Reset file pointer again
        file_data.seek(0)
        
        # Also try mimetypes as backup
        mime_type, _ = mimetypes.guess_type(file_data.name)
        
        return detected_format, mime_type or "application/octet-stream"
        
    except Exception as e:
        st.warning(f"Could not detect format for {file_data.name}: {str(e)}")
        return "UNKNOWN", "application/octet-stream"

def convert_to_png(file_data, quality: int = 95) -> Tuple[io.BytesIO, str]:
    """
    Convert an image to PNG format
    
    Args:
        file_data: StreamlitUploadedFile object
        quality: JPEG quality if converting from JPEG (1-100)
        
    Returns:
        Tuple of (converted_file_data, new_filename)
    """
    try:
        # Reset file pointer
        file_data.seek(0)
        
        # Open image with PIL
        with Image.open(file_data) as img:
            # Convert RGBA to RGB if necessary (PNG supports RGBA, but some formats don't)
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                img = background
            elif img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            
            # Create new filename with .png extension
            base_name = os.path.splitext(file_data.name)[0]
            new_filename = f"{base_name}.png"
            
            # Convert to PNG
            output_buffer = io.BytesIO()
            
            # Save as PNG with optimization
            img.save(output_buffer, format='PNG', optimize=True)
            output_buffer.seek(0)
            
            return output_buffer, new_filename
            
    except Exception as e:
        st.error(f"Failed to convert {file_data.name} to PNG: {str(e)}")
        # Return original file if conversion fails
        file_data.seek(0)
        return file_data, file_data.name

def process_uploaded_images(uploaded_files: List) -> List[Tuple[io.BytesIO, str, str, str]]:
    """
    Process uploaded images: detect format and convert to PNG if needed
    
    Args:
        uploaded_files: List of StreamlitUploadedFile objects
        
    Returns:
        List of tuples: (file_data, filename, original_format, was_converted)
    """
    processed_files = []
    
    for file in uploaded_files:
        # Detect format
        detected_format, mime_type = detect_image_format(file)
        
        # Check if conversion is needed
        needs_conversion = detected_format not in ['PNG', 'UNKNOWN']
        
        if needs_conversion:
            st.info(f"🔄 Converting {file.name} from {detected_format} to PNG...")
            converted_data, new_filename = convert_to_png(file)
            processed_files.append((converted_data, new_filename, detected_format, True))
            st.success(f"✅ Converted {file.name} → {new_filename}")
        else:
            # Reset file pointer for original file
            file.seek(0)
            processed_files.append((file, file.name, detected_format, False))
            st.info(f"ℹ️ {file.name} is already in {detected_format} format")
    
    return processed_files

def get_supported_formats() -> List[str]:
    """Get list of supported image formats"""
    return ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'gif', 'webp']

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def estimate_conversion_time(file_count: int, total_size_mb: float) -> str:
    """Estimate conversion time based on file count and size"""
    # Rough estimates: 0.1 seconds per file + 0.01 seconds per MB
    base_time = file_count * 0.1
    size_time = total_size_mb * 0.01
    total_seconds = base_time + size_time
    
    if total_seconds < 60:
        return f"~{total_seconds:.1f} seconds"
    elif total_seconds < 3600:
        minutes = total_seconds / 60
        return f"~{minutes:.1f} minutes"
    else:
        hours = total_seconds / 3600
        return f"~{hours:.1f} hours"
