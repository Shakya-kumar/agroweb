# AgroAlly - Smart Agriculture Assistant

AgroAlly is a comprehensive web-based agricultural assistant that helps farmers and agricultural enthusiasts make informed decisions about crop management, soil analysis, and plant disease detection.

## Features

### 1. Crop Recommendation System
- Analyzes soil parameters (N, P, K, temperature, humidity, pH, rainfall)
- Recommends suitable crops based on soil conditions
- Provides crop information in both English and Hindi

### 2. Soil Analysis
- Detailed analysis of soil parameters
- Provides fertilizer recommendations
- Offers specific treatment plans for soil improvement
- Includes general agricultural recommendations

### 3. Plant Disease Detection
- Upload plant images for disease analysis
- Uses machine learning to detect plant diseases
- Provides treatment recommendations
- Supports multiple crop types

### 4. User Management
- Secure user registration and login
- Personalized dashboard
- History tracking of all analyses and recommendations

## Technical Stack

- **Backend**: Python Flask
- **Database**: SQLite
- **Machine Learning**: PyTorch
- **Frontend**: HTML, CSS, Bootstrap
- **Image Processing**: OpenCV

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/Shakya-kumar/agroweb.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python login.py
```

4. Access the web application at `http://localhost:5000`

## Project Structure

- `login.py` - Main application file
- `templates/` - HTML templates
- `static/` - CSS, JavaScript, and uploads
- `models/` - Machine learning models
- `users.db` - SQLite database

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any queries or suggestions, please open an issue in the repository. 