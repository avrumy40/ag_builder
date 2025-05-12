# Deployment Guide for AG Hierarchy Builder

This guide provides instructions for deploying the AG Hierarchy Builder application to production environments.

## File Size Limit Configuration

The application supports large file uploads (up to 800MB). To ensure this works in your deployment environment, you need to:

### Option 1: Include the .streamlit/config.toml file

Make sure the `.streamlit/config.toml` file is included in your deployment. This file contains the necessary configuration:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
maxUploadSize = 800
```

### Option 2: Command Line Arguments

When starting the Streamlit server in your deployment environment, add the maxUploadSize parameter:

```bash
streamlit run app.py --server.maxUploadSize=800
```

### Option 3: Environment Configuration

For deployment platforms like Streamlit Cloud or similar services, you might need to set the configuration in their dashboard:

1. Look for "Advanced Settings" or "Configuration"
2. Set the `server.maxUploadSize` to `800`

## GitHub Pages Deployment

If deploying to GitHub Pages or similar static site hosting:

1. GitHub Pages doesn't support server-side applications directly, so you'll need a serverless approach
2. Consider using Streamlit Cloud for direct deployment from your GitHub repository
3. Set the maxUploadSize config parameter in the Streamlit Cloud settings

## Docker Deployment

If using Docker:

1. Add the following to your Dockerfile to ensure the config is included:
   ```dockerfile
   COPY .streamlit/config.toml /app/.streamlit/config.toml
   ```

2. Alternatively, add the command line parameter in your CMD/ENTRYPOINT:
   ```dockerfile
   CMD ["streamlit", "run", "app.py", "--server.maxUploadSize=800"]
   ```

## Checking Your Configuration

You can verify that the maxUploadSize setting is correctly applied by:

1. Uploading a large file (>200MB)
2. Checking the Streamlit server logs (should show the increased limit)