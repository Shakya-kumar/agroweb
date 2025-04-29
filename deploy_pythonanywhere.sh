#!/bin/bash

# Replace these variables with your PythonAnywhere username and app name
PYTHONANYWHERE_USERNAME="your_username"
APP_NAME="agroweb"

# Create necessary directories
mkdir -p static/uploads
mkdir -p models

# Install dependencies
pip install -r requirements.txt

# Initialize the database
python3 -c "
from login import init_db
init_db()
"

echo "Deployment script completed. Please follow these steps:"
echo "1. Go to https://www.pythonanywhere.com"
echo "2. Log in to your account"
echo "3. Go to the 'Web' tab"
echo "4. Click 'Add a new web app'"
echo "5. Choose 'Manual configuration' and 'Python 3.9'"
echo "6. In the 'Code' section, upload all your project files"
echo "7. In the 'Virtualenv' section, create a new virtualenv and install requirements"
echo "8. In the 'WSGI configuration file' section, make sure it points to your wsgi.py"
echo "9. Click 'Reload' to start your web app" 