import sys
import os

# Add your project directory to the Python path
path = '/home/Shakya-kumar/agroweb'
if path not in sys.path:
    sys.path.append(path)

# Set environment variables
os.environ['FLASK_APP'] = 'login.py'
os.environ['FLASK_ENV'] = 'production'

from login import app as application 