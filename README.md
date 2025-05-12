# AG Hierarchy Builder

A powerful tool for retail assortment planning and product grouping, supporting large datasets up to 800MB.

## Features

- Advanced attribute-based clustering techniques for product organization
- Interactive data visualization interfaces
- Support for large datasets (up to 800MB)
- Dynamic parameter tuning with intuitive controls
- Web-based analytical workflow for retail strategy

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements_for_deployment.txt
   ```

2. Run the application:
   ```
   ./run_app.sh
   ```

   This script sets up the proper configuration and starts the application with support for large file uploads.

## Deployment

For detailed deployment instructions, please see [DEPLOYMENT.md](DEPLOYMENT.md).

### Important: Large File Upload Support

This application is designed to handle large product catalogs (up to 800MB). When deploying:

1. **Always** make sure to include the `.streamlit/config.toml` file in your deployment
2. Or run with the command-line parameter: `--server.maxUploadSize=800`

## User Guides

- [Quick Reference Guide](AG_Hierarchy_Builder_Quick_Guide.md)
- [Complete User Guide](AG_Hierarchy_Builder_User_Guide.md)

## Troubleshooting

**Problem**: File uploads limited to 200MB in deployed environment

**Solution**: 
1. Make sure your deployment includes the `.streamlit/config.toml` file
2. Use the provided `run_app.sh` script to launch the application
3. For cloud platforms, add the parameter `--server.maxUploadSize=800` to your startup command

For more detailed troubleshooting and deployment options, see [DEPLOYMENT.md](DEPLOYMENT.md).