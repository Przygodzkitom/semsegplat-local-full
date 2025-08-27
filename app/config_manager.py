import os
import streamlit as st
from dotenv import load_dotenv, set_key
from pathlib import Path
import subprocess
import requests
import time

class ConfigManager:
    def __init__(self):
        self.env_path = Path('.env')
        self._load_environment()

    def _load_environment(self):
        """Load environment variables from .env file"""
        load_dotenv(self.env_path)

    def save_config(self, config_dict):
        """Save configuration to .env file"""
        for key, value in config_dict.items():
            set_key(self.env_path, key, value)
        self._load_environment()  # Reload environment

    def check_configuration(self):
        """Check if all required configuration is present"""
        required_vars = {
            "MINIO_BUCKET_NAME": "MinIO Bucket Name"
        }
        
        missing_vars = {
            key: name for key, name in required_vars.items() 
            if not os.getenv(key)
        }
        
        return missing_vars

    def setup_wizard(self):
        """Interactive setup wizard for first-time configuration"""
        st.title("⚙️ Project Setup")
        
        # MinIO Configuration
        st.subheader("☁️ MinIO Storage Configuration")
        with st.expander("How to set up MinIO", expanded=True):
            st.markdown("""
            1. MinIO is automatically started with Docker Compose
            2. Default bucket: segmentation-platform
            3. Access console at: http://localhost:9001
            4. Login: minioadmin / minioadmin123
            """)
        
        minio_bucket = st.text_input(
            "MinIO Bucket Name",
            value=os.getenv("MINIO_BUCKET_NAME", "segmentation-platform"),
            help="The name of your MinIO bucket"
        )
        
        if minio_bucket:
            self.save_config({"MINIO_BUCKET_NAME": minio_bucket})
            st.success("✅ MinIO bucket name saved!")
        
        # Check if all configuration is complete
        missing_vars = self.check_configuration()
        if missing_vars:
            st.warning("Please complete all configuration fields above.")
            return False
            
        return True

def get_config_manager():
    """Get or create a configuration manager instance"""
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    return st.session_state.config_manager 