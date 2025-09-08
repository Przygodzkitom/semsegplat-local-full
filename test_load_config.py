import sys
import os
sys.path.append('app')

# Mock streamlit session state
class MockSessionState:
    def __init__(self):
        self.data = {}
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def __contains__(self, key):
        return key in self.data

# Mock streamlit
class MockStreamlit:
    def __init__(self):
        self.session_state = MockSessionState()
    
    def success(self, msg):
        print(f"SUCCESS: {msg}")
    
    def error(self, msg):
        print(f"ERROR: {msg}")
    
    def warning(self, msg):
        print(f"WARNING: {msg}")

# Replace streamlit module
sys.modules['streamlit'] = MockStreamlit()

# Now import and test the function
from main import load_project_config

print("Testing load_project_config function...")
result = load_project_config()
print(f"Result: {result}")
print(f"Session state: {MockStreamlit().session_state.data}")


