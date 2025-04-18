# Crop Prediction System Backend

This is the backend API for the Crop Prediction System, which provides crop recommendations based on soil and environmental conditions.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the `Crop_dataset.csv` file is in the same directory as `app.py`

3. Run the Flask application:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### 1. Get Crop Recommendation
- **Endpoint**: `/api/recommend`
- **Method**: POST
- **Request Body**:
```json
{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
}
```
- **Response**:
```json
{
    "recommended_crop": "rice",
    "hindi_name": "चावल",
    "details": {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
        "label": "rice"
    }
}
```

### 2. Get Crop Requirements
- **Endpoint**: `/api/crop-requirements`
- **Method**: POST
- **Request Body**:
```json
{
    "crop_name": "rice",
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
}
```
- **Response**:
```json
{
    "crop": "rice",
    "hindi_name": "चावल",
    "requirements": [
        {
            "parameter": "N",
            "current_value": 90.0,
            "required_value": 95.0,
            "adjustment": 5.0,
            "unit": "kg/ha"
        },
        // ... other parameters
    ]
}
```

## Error Handling

The API returns appropriate error messages with HTTP status codes:
- 400: Bad Request (invalid input data)
- 500: Internal Server Error

## Notes

- All numerical values should be provided as floats
- pH values must be between 1 and 14
- Crop names can be provided in either English or Hindi 