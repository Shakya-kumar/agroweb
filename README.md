# AgroAlly - Smart Farming Solutions

AgroAlly is an AI-powered smart farming platform that helps farmers make better agricultural decisions through advanced data analysis and machine learning.

## Features

- **Crop Recommendation**: Get personalized crop recommendations based on soil parameters and environmental conditions
- **Soil Analysis**: Analyze soil health and receive detailed improvement suggestions
- **Disease Detection**: AI-powered plant disease detection and treatment recommendations
- **History Tracking**: Monitor and track all your farming analyses and recommendations

## Technologies Used

- Python 3.8+
- Flask
- SQLite
- PyTorch
- Bootstrap 5
- Font Awesome
- Machine Learning & AI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agroally.git
cd agroally
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python login.py
```

5. Run the application:
```bash
flask run
```

The application will be available at `http://localhost:5000`

## Project Structure

- `login.py`: Main application file with Flask routes and database setup
- `templates/`: HTML templates using Jinja2
- `static/`: Static files (CSS, JavaScript, images)
- `models/`: Machine learning models
- `dataset/`: Training datasets
- `requirements.txt`: Python dependencies

## Environment Variables

Create a `.env` file in the root directory with the following variables:
```
FLASK_APP=login.py
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Email: agroally@gmail.com
- Website: [agroally.com](https://agroally.com)

## Acknowledgments

- Thanks to all contributors who have helped shape AgroAlly
- Special thanks to the agricultural research community for providing valuable datasets
- Icons provided by Font Awesome 