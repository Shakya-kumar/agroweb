from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL)''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Load crop dataset
df = pd.read_csv('Crop_dataset.csv')

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

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# Function to find the closest row in the dataset
def find_closest_row(input_values, dataset):
    distances = dataset.iloc[:, :-2].apply(lambda row: euclidean_distance(input_values, row), axis=1)
    closest_index = distances.idxmin()
    closest_row = dataset.loc[closest_index]
    return closest_row

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

if __name__ == '__main__':
    app.run(debug=True) 