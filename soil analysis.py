print("------------------------------------Soil Analysis----------------------------------------------")
import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('Crop_dataset.csv')

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Function to validate and get input for pH (should be within 1 to 14)
def get_valid_ph():
    while True:
        try:
            ph = float(input("Enter the value of pH (1 to 14): "))
            if 1 <= ph <= 14:
                return ph
            else:
                print("Error: pH value must be between 1 and 14. Please try again.")
        except ValueError:
            print("Error: Invalid input. Please enter a valid number.")

# Function to find the closest crop based on input values
def find_closest_crop(input_values, dataset):
    closest_crop = None
    min_distance = float('inf')

    for _, row in dataset.iterrows():
        row_values = row[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].values
        distance = euclidean_distance(input_values, row_values)
        if distance < min_distance:
            min_distance = distance
            closest_crop = row['label']

    return closest_crop

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
    "Watermelon": "तरबूज",
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

# Take inputs for soil attributes
attributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
input_values = []

print("Enter the values for nitrogen composition (N - kg/ha), phosphorus content (P - kg/ha), potassium content (K - kg/ha), temperature (°C), humidity (%), pH, and rainfall (mm):")

for attr in attributes:
    if attr == "ph":
        value = get_valid_ph()
    else:
        unit = "kg/ha" if attr in ["N", "P", "K"] else "°C" if attr == "temperature" else "%" if attr == "humidity" else "mm"
        value = float(input(f"Enter the value of {attr} ({unit}): "))
    input_values.append(value)

input_values = np.array(input_values)

# Find the closest crop in the dataset
closest_crop = find_closest_crop(input_values, df)

# Determine the Hindi name
hindi_name = crop_name_mapping.get(closest_crop, "Unknown")

# Print the closest matching crop with its Hindi name
print(f"Based on your input, your soil is suitable to grow {closest_crop} ({hindi_name}).")

# Calculate adjustments needed based on the differences
crop_data = df[df['label'] == closest_crop]
avg_values = crop_data[attributes].mean().values
adjustments = avg_values - input_values

# Print adjustments needed for each attribute
print(f"To grow {closest_crop} ({hindi_name}), you need the following adjustments:")
for attr, adjustment in zip(attributes, adjustments):
    current_value = input_values[attributes.index(attr)]
    required_value = avg_values[attributes.index(attr)]

    if adjustment > 0:
        print(f"Your {attr} is {current_value:.2f}, but {closest_crop} ({hindi_name}) requires an average of {required_value:.2f}.")
        print(f"You need to boost {attr} by {adjustment:.2f}.")
    elif adjustment < 0:
        print(f"Your {attr} is {current_value:.2f}, but {closest_crop} ({hindi_name}) requires an average of {required_value:.2f}.")
        print(f"You have excess {attr} by {-adjustment:.2f}.")
    else:
        print(f"Your {attr} is already optimal for {closest_crop} ({hindi_name}).")

print("Thank you for using the program!")