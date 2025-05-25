# Guide to Hosting Your Streamlit App on Render

This guide will walk you through the process of deploying your Air Quality Prediction System on Render.

## Prerequisites

1. A GitHub account
2. Your project code pushed to a GitHub repository
3. A Render account (sign up at [render.com](https://render.com))

## Step 1: Prepare Your Repository

Ensure your repository has the following files:

1. All your Python code files (app.py, data_fetcher.py, etc.)
2. A `requirements.txt` file (use the contents from your streamlit_requirements.txt)
3. A `render.yaml` file (optional, for advanced configuration)

## Step 2: Create a requirements.txt File

If you haven't already, create a requirements.txt file with the following contents:

```
matplotlib==3.10.3
numpy==2.2.6
pandas==2.2.3
plotly==6.1.1
requests==2.32.3
scikit-learn==1.6.1
seaborn==0.13.2
streamlit==1.45.1
trafilatura==2.0.0
xgboost==3.0.2
```

## Step 3: Create a start.sh File

Create a `start.sh` file in your repository with the following content:

```bash
#!/bin/bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Make sure to make this file executable if you're using Git Bash or a Unix-like system:

```bash
chmod +x start.sh
```

## Step 4: Set Up Deployment on Render

1. Log in to your Render account
2. Click on "New" and select "Web Service"
3. Connect your GitHub repository
4. Configure your web service with the following settings:
   - **Name**: Choose a name for your app (e.g., air-quality-prediction)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `sh start.sh`
   - **Instance Type**: Free (or select a paid plan for better performance)

## Step 5: Environment Variables (if needed)

If your app uses API keys or other secrets:

1. Go to the "Environment" tab in your Render dashboard
2. Add your environment variables (e.g., API keys)
3. Click "Save Changes"

## Step 6: Deploy

1. Click "Create Web Service"
2. Wait for the build and deployment process to complete
3. Once deployed, Render will provide you with a URL to access your app

## Troubleshooting

If you encounter issues during deployment:

1. Check the build logs in the Render dashboard
2. Ensure all dependencies are correctly listed in requirements.txt
3. Verify that your app works locally before deploying
4. Make sure your app is configured to listen on the port provided by Render's $PORT environment variable

## Updating Your App

To update your deployed app:

1. Push changes to your GitHub repository
2. Render will automatically detect the changes and rebuild your app
3. Monitor the build logs to ensure successful deployment

## Additional Resources

- [Render Documentation](https://render.com/docs)
- [Streamlit Deployment Guide](https://docs.streamlit.io/knowledge-base/deploy/deploying-streamlit-app)