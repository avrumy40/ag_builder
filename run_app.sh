#!/bin/bash

# Run the configuration helper
echo "Setting up Streamlit configuration..."
python streamlit_config.py

# Start the Streamlit app with the required parameters
echo "Starting AG Hierarchy Builder..."
streamlit run app.py --server.maxUploadSize=800