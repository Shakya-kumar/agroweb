import sys
import os

# Add your project directory to the Python path
path = os.path.dirname(os.path.abspath(__file__))
if path not in sys.path:
    sys.path.append(path)

# Set environment variables
os.environ['FLASK_APP'] = 'login.py'
os.environ['FLASK_ENV'] = 'production'
os.environ['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key

# Create necessary directories
os.makedirs(os.path.join(path, 'static/uploads'), exist_ok=True)
os.makedirs(os.path.join(path, 'models'), exist_ok=True)

from login import app as application 