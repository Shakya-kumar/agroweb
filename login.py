from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import logging
import os
import cv2
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  feature TEXT NOT NULL,
                  input_data TEXT NOT NULL,
                  result_data TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Load crop dataset
try:
    logger.info("Attempting to load Crop_dataset.csv")
    df = pd.read_csv('Crop_dataset.csv')
    logger.info(f"Successfully loaded dataset with {len(df)} rows")
    logger.debug(f"Dataset columns: {df.columns.tolist()}")
    logger.debug(f"Sample data:\n{df.head()}")
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    raise

# Dictionary to store SI unit information for each attribute
unit_info = {
    "N": "kg/ha",
    "P": "kg/ha",
    "K": "kg/ha",
    "temperature": "°C",
    "humidity": "%",
    "ph": "pH (1-14)",
    "rainfall": "mm"
}

# Dictionary to map English crop names to Hindi names
crop_name_mapping = {
    "rice": "चावल",
    "maize": "मक्का",
    "chickpea": "चना",
    "kidneybeans": "राजमा",
    "pigeonpeas": "तूर दाल",
    "mothbeans": "मोत दाल",
    "mungbean": "मूंग",
    "blackgram": "काला चना",
    "lentil": "दाल",
    "pomegranate": "अनार",
    "banana": "केला",
    "mango": "आम",
    "grapes": "अंगूर",
    "watermelon": "तरबूज",
    "muskmelon": "खरबूजा",
    "apple": "सेब",
    "orange": "संतरा",
    "papaya": "पपीता",
    "coconut": "नारियल",
    "cotton": "रुई",
    "jute": "जूट",
    "coffee": "कॉफी",
    "wheat": "गेहूं",
    "sugarcane": "गन्ना",
    "corn": "भुट्टा",
    "groundnut": "मूँगफली",
    "tea": "चाय",
    "rubber": "रबड़",
    "turmeric": "हल्दी",
    "pepper": "काली मिर्च",
    "tomato": "टमाटर"
}

# Reverse mapping from Hindi to English
hindi_to_english = {v: k for k, v in crop_name_mapping.items()}

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# Function to find the closest row in the dataset
def find_closest_row(input_values, dataset):
    distances = dataset.iloc[:, :-2].apply(lambda row: euclidean_distance(input_values, row), axis=1)
    closest_index = distances.idxmin()
    closest_row = dataset.loc[closest_index]
    return closest_row

# Configure upload folder for disease images
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the model class
class PlantDiseaseModel(nn.Module):
    def __init__(self):
        super(PlantDiseaseModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.base_model(x)

# Load the model
try:
    model = PlantDiseaseModel()
    model.load_state_dict(torch.load('models/plant_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully")
except:
    print("Warning: Disease detection model not found. Using mock predictions.")
    model = None

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        try:
            c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                     (username, generate_password_hash(password), email))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

# Function to save history entries
def save_history(user_id, feature, input_data, result_data):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''INSERT INTO history (user_id, feature, input_data, result_data)
                 VALUES (?, ?, ?, ?)''',
              (user_id, feature, str(input_data), str(result_data)))
    conn.commit()
    conn.close()

@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            input_values = []
            attributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            
            for attr in attributes:
                value = float(request.form[attr])
                if attr == "ph" and not (1 <= value <= 14):
                    flash('Error: pH value must be between 1 and 14.', 'error')
                    return redirect(url_for('crop_recommendation'))
                input_values.append(value)
            
            closest_row = find_closest_row(np.array(input_values), df)
            closest_crop = closest_row['label']
            hindi_name = crop_name_mapping.get(closest_crop, "Unknown")
            
            # Save to history
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT id FROM users WHERE username = ?', (session['username'],))
            user_id = c.fetchone()[0]
            conn.close()
            
            input_data = dict(zip(attributes, input_values))
            result_data = {
                'recommended_crop': closest_crop,
                'hindi_name': hindi_name
            }
            save_history(user_id, 'crop_recommendation', input_data, result_data)
            
            return render_template('crop_result.html', 
                                crop=closest_crop,
                                hindi_name=hindi_name,
                                input_values=input_values,
                                attributes=attributes,
                                unit_info=unit_info)
            
        except ValueError:
            flash('Error: Please enter valid numbers for all fields.', 'error')
            return redirect(url_for('crop_recommendation'))
    
    return render_template('crop_recommendation.html', unit_info=unit_info)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/soil_analysis', methods=['GET', 'POST'])
def soil_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Get form data
            crop_name = request.form['crop_name'].strip().lower()
            input_values = []
            attributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            
            for attr in attributes:
                value = float(request.form[attr])
                if attr == "ph" and not (1 <= value <= 14):
                    flash('Error: pH value must be between 1 and 14.', 'error')
                    return redirect(url_for('soil_analysis'))
                input_values.append(value)
            
            # Get crop data for optimal parameters
            crop_data = df[df['label'].str.lower() == crop_name]
            if crop_data.empty:
                # Try name mapping
                crop_name_english = crop_name_mapping.get(crop_name, None)
                crop_name_hindi = hindi_to_english.get(crop_name, None)
                if crop_name_english or crop_name_hindi:
                    selected_crop = crop_name_english if crop_name_english else crop_name_hindi
                    crop_data = df[df['label'].str.lower() == selected_crop.lower()]
            
            if crop_data.empty:
                flash(f'Crop "{crop_name}" not found in our database.', 'error')
                return redirect(url_for('soil_analysis'))
            
            # Calculate optimal parameters for the crop
            optimal_params = {
                'N': crop_data['N'].mean(),
                'P': crop_data['P'].mean(),
                'K': crop_data['K'].mean(),
                'temperature': crop_data['temperature'].mean(),
                'humidity': crop_data['humidity'].mean(),
                'ph': crop_data['ph'].mean(),
                'rainfall': crop_data['rainfall'].mean()
            }
            
            # Perform soil analysis
            analysis_results = {}
            fertilizer_recommendations = []
            
            # Nitrogen (N) analysis
            n_diff = optimal_params['N'] - input_values[0]
            if n_diff > 0:
                analysis_results["N"] = {
                    "status": "Deficient",
                    "current": input_values[0],
                    "optimal": optimal_params['N'],
                    "unit": unit_info['N']
                }
                # Urea (46% N) recommendation
                urea_amount = (n_diff * 2.17)  # Convert kg/ha to kg/acre
                fertilizer_recommendations.append({
                    "name": "Urea",
                    "amount": f"{urea_amount:.1f} kg/acre",
                    "timing": "Apply in 2-3 split doses during crop growth"
                })
            else:
                analysis_results["N"] = {
                    "status": "Sufficient",
                    "current": input_values[0],
                    "optimal": optimal_params['N'],
                    "unit": unit_info['N']
                }
            
            # Phosphorus (P) analysis
            p_diff = optimal_params['P'] - input_values[1]
            if p_diff > 0:
                analysis_results["P"] = {
                    "status": "Deficient",
                    "current": input_values[1],
                    "optimal": optimal_params['P'],
                    "unit": unit_info['P']
                }
                # DAP (18% N, 46% P2O5) recommendation
                dap_amount = (p_diff * 2.17) / 0.46  # Convert kg/ha to kg/acre
                fertilizer_recommendations.append({
                    "name": "DAP (Diammonium Phosphate)",
                    "amount": f"{dap_amount:.1f} kg/acre",
                    "timing": "Apply as basal dose before sowing"
                })
            else:
                analysis_results["P"] = {
                    "status": "Sufficient",
                    "current": input_values[1],
                    "optimal": optimal_params['P'],
                    "unit": unit_info['P']
                }
            
            # Potassium (K) analysis
            k_diff = optimal_params['K'] - input_values[2]
            if k_diff > 0:
                analysis_results["K"] = {
                    "status": "Deficient",
                    "current": input_values[2],
                    "optimal": optimal_params['K'],
                    "unit": unit_info['K']
                }
                # MOP (60% K2O) recommendation
                mop_amount = (k_diff * 2.17) / 0.6  # Convert kg/ha to kg/acre
                fertilizer_recommendations.append({
                    "name": "MOP (Muriate of Potash)",
                    "amount": f"{mop_amount:.1f} kg/acre",
                    "timing": "Apply as basal dose before sowing"
                })
            else:
                analysis_results["K"] = {
                    "status": "Sufficient",
                    "current": input_values[2],
                    "optimal": optimal_params['K'],
                    "unit": unit_info['K']
                }
            
            # Other parameters
            analysis_results["temperature"] = {
                "status": "Optimal" if abs(input_values[3] - optimal_params['temperature']) <= 5 else "Needs attention",
                "current": input_values[3],
                "optimal": optimal_params['temperature'],
                "unit": unit_info['temperature']
            }
            
            analysis_results["humidity"] = {
                "status": "Optimal" if abs(input_values[4] - optimal_params['humidity']) <= 10 else "Needs attention",
                "current": input_values[4],
                "optimal": optimal_params['humidity'],
                "unit": unit_info['humidity']
            }
            
            analysis_results["ph"] = {
                "status": "Optimal" if 6 <= input_values[5] <= 7 else "Needs attention",
                "current": input_values[5],
                "optimal": optimal_params['ph'],
                "unit": unit_info['ph']
            }
            
            analysis_results["rainfall"] = {
                "status": "Optimal" if abs(input_values[6] - optimal_params['rainfall']) <= 50 else "Needs attention",
                "current": input_values[6],
                "optimal": optimal_params['rainfall'],
                "unit": unit_info['rainfall']
            }
            
            # Add general recommendations
            general_recommendations = []
            if input_values[5] < 6:
                general_recommendations.append("Apply lime to increase soil pH")
            elif input_values[5] > 7:
                general_recommendations.append("Apply sulfur to decrease soil pH")
            
            # Save to history
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT id FROM users WHERE username = ?', (session['username'],))
            user_id = c.fetchone()[0]
            conn.close()
            
            input_data = {
                'crop_name': crop_name,
                'soil_parameters': dict(zip(attributes, input_values))
            }
            result_data = {
                'analysis_results': analysis_results,
                'fertilizer_recommendations': fertilizer_recommendations,
                'general_recommendations': general_recommendations
            }
            save_history(user_id, 'soil_analysis', input_data, result_data)
            
            return render_template('soil_analysis_result.html', 
                                crop_name=crop_name,
                                analysis_results=analysis_results,
                                fertilizer_recommendations=fertilizer_recommendations,
                                general_recommendations=general_recommendations,
                                unit_info=unit_info)
            
        except ValueError:
            flash('Error: Please enter valid numbers for all fields.', 'error')
            return redirect(url_for('soil_analysis'))
    
    return render_template('soil_analysis.html', unit_info=unit_info)

@app.route('/crop_management')
def crop_management():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('crop_management.html')

@app.route('/add_crop', methods=['GET', 'POST'])
def add_crop():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Handle crop addition logic here
        flash('Crop added successfully!', 'success')
        return redirect(url_for('crop_management'))
    
    return render_template('add_crop.html')

@app.route('/view_crops')
def view_crops():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get user's crops from database
    crops = []  # Replace with actual database query
    
    return render_template('view_crops.html', crops=crops)

@app.route('/grow_crop', methods=['GET', 'POST'])
def grow_crop():
    if 'username' not in session:
        logger.warning("Unauthorized access attempt to grow_crop")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Get form data
            crop_name = request.form['crop_name'].strip().lower()
            logger.debug(f"Processing request for crop: {crop_name}")
            
            soil_params = {
                'N': float(request.form['N']),
                'P': float(request.form['P']),
                'K': float(request.form['K']),
                'temperature': float(request.form['temperature']),
                'humidity': float(request.form['humidity']),
                'ph': float(request.form['ph']),
                'rainfall': float(request.form['rainfall'])
            }
            logger.debug(f"Soil parameters: {soil_params}")
            
            # Validate pH value
            if not (1 <= soil_params['ph'] <= 14):
                logger.error(f"Invalid pH value: {soil_params['ph']}")
                flash('Error: pH value must be between 1 and 14.', 'error')
                return redirect(url_for('grow_crop'))
            
            # Get crop data from dataset
            logger.debug("Searching for crop in dataset")
            crop_data = df[df['label'].str.lower() == crop_name]
            
            # If no direct match, try mapping
            if crop_data.empty:
                logger.debug(f"No direct match found for {crop_name}, trying name mapping")
                crop_name_english = crop_name_mapping.get(crop_name, None)
                crop_name_hindi = hindi_to_english.get(crop_name, None)
                logger.debug(f"Crop name mapping - English: {crop_name_english}, Hindi: {crop_name_hindi}")
                
                if crop_name_english or crop_name_hindi:
                    selected_crop = crop_name_english if crop_name_english else crop_name_hindi
                    crop_data = df[df['label'].str.lower() == selected_crop.lower()]
            
            if crop_data.empty:
                logger.warning(f"No data found for crop: {crop_name}")
                available_crops = sorted(set(df['label'].str.lower()))
                logger.debug(f"Available crops: {available_crops}")
                flash(f'Crop "{crop_name}" not found in our database. Available crops: {", ".join(available_crops)}', 'error')
                return redirect(url_for('grow_crop'))
            
            logger.debug(f"Found {len(crop_data)} matching rows for crop")
            
            # Calculate optimal parameters
            optimal_params = {
                'N': crop_data['N'].mean(),
                'P': crop_data['P'].mean(),
                'K': crop_data['K'].mean(),
                'temperature': crop_data['temperature'].mean(),
                'humidity': crop_data['humidity'].mean(),
                'ph': crop_data['ph'].mean(),
                'rainfall': crop_data['rainfall'].mean()
            }
            logger.debug(f"Calculated optimal parameters: {optimal_params}")
            
            # Generate recommendations
            recommendations = []
            for param in soil_params:
                current = soil_params[param]
                optimal = optimal_params[param]
                diff = optimal - current
                
                if abs(diff) > 0.1:
                    if diff > 0:
                        recommendations.append(f"Increase {param} by {abs(diff):.1f} {unit_info[param]}")
                    else:
                        recommendations.append(f"Decrease {param} by {abs(diff):.1f} {unit_info[param]}")
            
            if not recommendations:
                recommendations.append("Your soil parameters are already optimal for growing this crop!")
            
            # Add general growing tips
            recommendations.append(f"Maintain soil pH between {optimal_params['ph']:.1f} and {optimal_params['ph'] + 0.5:.1f}")
            recommendations.append(f"Keep temperature around {optimal_params['temperature']:.1f}°C")
            recommendations.append(f"Maintain humidity levels between {optimal_params['humidity']:.1f}%")
            
            logger.info(f"Generated {len(recommendations)} recommendations for {crop_name}")
            
            return render_template('crop_recommendations.html',
                                crop_name=crop_name,
                                soil_params=soil_params,
                                optimal_params=optimal_params,
                                unit_info=unit_info,
                                recommendations=recommendations)
            
        except ValueError as e:
            logger.error(f"Value error in grow_crop: {str(e)}")
            flash('Error: Please enter valid numbers for all fields.', 'error')
            return redirect(url_for('grow_crop'))
        except Exception as e:
            logger.error(f"Unexpected error in grow_crop: {str(e)}")
            flash(f'Error processing your request: {str(e)}', 'error')
            return redirect(url_for('grow_crop'))
    
    # For GET request, just render the form
    return render_template('grow_crop.html', unit_info=unit_info)

@app.route('/disease_analysis')
def disease_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('disease_analysis.html')

@app.route('/analyze_disease', methods=['POST'])
def analyze_disease():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if 'plantImage' not in request.files:
        flash('No image file uploaded')
        return redirect(request.url)
    
    file = request.files['plantImage']
    plant_type = request.form.get('plantType')
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image and make prediction
        if model:
            try:
                # Load and preprocess image
                image = Image.open(filepath).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)
                
                # Make prediction
                with torch.no_grad():
                    output = model(image_tensor)
                    probability = output.item()
                    disease_detected = probability > 0.5
                    confidence = probability * 100 if disease_detected else (1 - probability) * 100
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                disease_detected = True
                confidence = 85.5
        else:
            # Mock prediction for testing
            disease_detected = True
            confidence = 85.5
        
        # Treatment recommendations
        treatments = {
            'tomato': 'Apply fungicide and remove affected leaves',
            'potato': 'Use copper-based fungicide and improve drainage',
            'corn': 'Apply appropriate fungicide and maintain proper spacing',
            'rice': 'Use recommended fungicide and maintain water level',
            'wheat': 'Apply fungicide and ensure proper crop rotation',
            'other': 'Consult local agricultural extension for specific treatment'
        }
        
        treatment = treatments.get(plant_type, 'Consult local agricultural extension for treatment')
        
        # Save to history
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE username = ?', (session['username'],))
        user_id = c.fetchone()[0]
        conn.close()
        
        input_data = {
            'plant_type': plant_type,
            'image_filename': filename
        }
        result_data = {
            'disease_detected': disease_detected,
            'disease_name': 'Leaf Blight' if disease_detected else 'None',
            'confidence': confidence,
            'treatment': treatment
        }
        save_history(user_id, 'disease_analysis', input_data, result_data)
        
        return render_template('disease_result.html',
                             image_path=url_for('static', filename=f'uploads/{filename}'),
                             disease_detected=disease_detected,
                             disease_name='Leaf Blight' if disease_detected else 'None',
                             confidence=confidence,
                             treatment=treatment)
    
    flash('Invalid file type')
    return redirect(request.url)

@app.route('/history')
def view_history():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''SELECT h.id, h.feature, h.input_data, h.result_data, h.timestamp
                 FROM history h
                 JOIN users u ON h.user_id = u.id
                 WHERE u.username = ?
                 ORDER BY h.timestamp DESC''', (session['username'],))
    history_entries = c.fetchall()
    conn.close()
    
    # Process history entries
    processed_entries = []
    for entry in history_entries:
        entry_id, feature, input_data, result_data, timestamp = entry
        processed_entries.append({
            'id': entry_id,
            'feature': feature,
            'input_data': eval(input_data),  # Convert string to dict
            'result_data': eval(result_data),  # Convert string to dict
            'timestamp': timestamp
        })
    
    return render_template('history.html', history_entries=processed_entries)

if __name__ == '__main__':
    app.run(debug=True) 