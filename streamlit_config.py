"""
This script checks and sets up the proper Streamlit configuration.
Run this before starting your Streamlit app to ensure proper configuration.
"""

import os
import toml
import sys

# Default configuration directory
CONFIG_DIR = '.streamlit'
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.toml')

def ensure_config():
    """Ensures that the Streamlit config file exists with the proper settings."""
    # Create config directory if it doesn't exist
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        print(f"Created directory: {CONFIG_DIR}")
    
    # Read existing config if it exists
    config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = toml.load(f)
            print(f"Loaded existing config from {CONFIG_FILE}")
        except Exception as e:
            print(f"Error reading config file: {e}")
    
    # Ensure server section exists
    if 'server' not in config:
        config['server'] = {}
    
    # Set required parameters
    config['server']['headless'] = True
    config['server']['address'] = "0.0.0.0"
    config['server']['port'] = 5000
    config['server']['maxUploadSize'] = 800
    
    # Write config back to file
    try:
        with open(CONFIG_FILE, 'w') as f:
            toml.dump(config, f)
        print(f"Updated config in {CONFIG_FILE}")
        print(f"Set maxUploadSize to 800 MB")
    except Exception as e:
        print(f"Error writing config file: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Checking Streamlit configuration...")
    success = ensure_config()
    if success:
        print("Streamlit configuration is ready!")
    else:
        print("Failed to configure Streamlit!")
        sys.exit(1)