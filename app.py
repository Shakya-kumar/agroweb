from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

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

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def find_closest_row(input_values, dataset):
    try:
        distances = dataset.iloc[:, :-2].apply(lambda row: euclidean_distance(input_values, row), axis=1)
        closest_index = distances.idxmin()
        closest_row = dataset.loc[closest_index]
        return closest_row
    except Exception as e:
        logger.error(f"Error in find_closest_row: {str(e)}")
        raise

def get_crop_recommendations(input_values):
    try:
        # Read the dataset
        logger.info("Reading Crop_dataset.csv")
        df = pd.read_csv('Crop_dataset.csv')
        
        # Log input values for debugging
        logger.debug(f"Input values: {input_values}")
        
        # Find the closest row
        closest_row = find_closest_row(np.array(input_values), df)
        closest_crop = closest_row['label']
        hindi_name = crop_name_mapping.get(closest_crop, "Unknown")
        
        logger.info(f"Recommended crop: {closest_crop} ({hindi_name})")
        
        return {
            "recommended_crop": closest_crop,
            "hindi_name": hindi_name,
            "details": closest_row.to_dict()
        }
    except Exception as e:
        logger.error(f"Error in get_crop_recommendations: {str(e)}")
        raise

def get_crop_requirements(crop_name, input_values):
    try:
        # Read the dataset
        logger.info("Reading Crop_dataset.csv for requirements")
        df = pd.read_csv('Crop_dataset.csv')
        logger.debug(f"Dataset loaded with {len(df)} rows")
        
        # Convert crop name to lowercase for case-insensitive comparison
        crop_name = crop_name.lower().strip()
        logger.debug(f"Looking for crop (lowercase): {crop_name}")
        
        # First try direct match in dataset
        crop_data = df[df['label'].str.lower() == crop_name]
        
        # If no direct match, try mapping
        if crop_data.empty:
            # Check if crop exists in either language
            crop_name_english = crop_name_mapping.get(crop_name, None)
            crop_name_hindi = hindi_to_english.get(crop_name, None)
            logger.debug(f"Crop name mapping - English: {crop_name_english}, Hindi: {crop_name_hindi}")
            
            if crop_name_english or crop_name_hindi:
                selected_crop = crop_name_english if crop_name_english else crop_name_hindi
                crop_data = df[df['label'].str.lower() == selected_crop.lower()]
        
        # If still no match, try fuzzy matching
        if crop_data.empty:
            # Try to find similar crop names
            all_crops = df['label'].unique()
            logger.debug(f"Available crops in dataset: {all_crops}")
            logger.warning(f"No exact match found for crop: {crop_name}")
            return {"error": f"No data available for crop: {crop_name}. Available crops: {', '.join(sorted(set(df['label'].str.lower())))}"}
        
        logger.debug(f"Found {len(crop_data)} rows for crop")
        
        # Calculate average values and adjustments
        attributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        avg_values = crop_data[attributes].mean().values
        adjustments = avg_values - np.array(input_values)
        
        # Prepare response
        requirements = []
        for attr, adjustment in zip(attributes, adjustments):
            current_value = input_values[attributes.index(attr)]
            required_value = avg_values[attributes.index(attr)]
            
            requirements.append({
                "parameter": attr,
                "current_value": float(current_value),
                "required_value": float(required_value),
                "adjustment": float(adjustment),
                "unit": unit_info[attr]
            })
        
        # Get the actual crop name from the dataset for consistent casing
        actual_crop_name = crop_data['label'].iloc[0]
        logger.info(f"Requirements calculated for {actual_crop_name}")
        return {
            "crop": actual_crop_name,
            "hindi_name": crop_name_mapping.get(actual_crop_name.lower(), "Unknown"),
            "requirements": requirements
        }
    except Exception as e:
        logger.error(f"Error in get_crop_requirements: {str(e)}")
        raise

@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/recommend', methods=['POST'])
def recommend_crop():
    try:
        data = request.get_json()
        logger.debug(f"Received recommendation request: {data}")
        
        # Validate input data
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        input_values = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]
        
        # Validate pH value
        if not (1 <= input_values[5] <= 14):
            logger.error(f"Invalid pH value: {input_values[5]}")
            return jsonify({"error": "pH value must be between 1 and 14"}), 400
            
        result = get_crop_recommendations(input_values)
        return jsonify(result)
        
    except ValueError as e:
        logger.error(f"Value error in recommend_crop: {str(e)}")
        return jsonify({"error": "Invalid input values. Please ensure all fields are numbers."}), 400
    except Exception as e:
        logger.error(f"Error in recommend_crop: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/crop-requirements', methods=['POST'])
def crop_requirements():
    try:
        data = request.get_json()
        logger.debug(f"Received requirements request with data: {data}")
        
        # Validate input data
        required_fields = ['crop_name', 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        crop_name = data['crop_name']
        logger.debug(f"Processing request for crop: {crop_name}")
        
        input_values = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]
        logger.debug(f"Input values: {input_values}")
        
        # Validate pH value
        if not (1 <= input_values[5] <= 14):
            logger.error(f"Invalid pH value: {input_values[5]}")
            return jsonify({"error": "pH value must be between 1 and 14"}), 400
            
        result = get_crop_requirements(crop_name, input_values)
        logger.debug(f"Requirements result: {result}")
        return jsonify(result)
        
    except ValueError as e:
        logger.error(f"Value error in crop_requirements: {str(e)}")
        return jsonify({"error": "Invalid input values. Please ensure all fields are numbers."}), 400
    except Exception as e:
        logger.error(f"Error in crop_requirements: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 