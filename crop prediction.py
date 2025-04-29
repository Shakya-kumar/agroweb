print("---------------------------------------------- Crop Prediction -------------------------------------------")
import pandas as pd
import numpy as np

# Read the CSV file (replace 'Crop_dataset.csv' with the correct file path)
df = pd.read_csv('Crop_dataset.csv')

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# Function to find the closest row in the dataset
def find_closest_row(input_values, dataset):
    distances = dataset.iloc[:, :-2].apply(lambda row: euclidean_distance(input_values, row), axis=1)
    closest_index = distances.idxmin()
    closest_row = dataset.loc[closest_index]
    return closest_row

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

# Take input values in SI units
print("Enter the values for nitrogen composition (N), Phosphorus content (P), potassium content (K),")
print("temperature (in degree Celsius), humidity (%), pH (in the range of 1 to 14), rainfall (in mm)")

attributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
input_values = []

# Collect user input and convert to appropriate data type
for attr in attributes:
    unit = unit_info[attr]
    while True:
        try:
            value = float(input(f"Enter the value of {attr} ({unit}): "))
            if attr == "ph" and not (1 <= value <= 14):
                print("Error: pH value must be between 1 and 14.")
                continue
            input_values.append(value)
            break
        except ValueError:
            print("Error: Invalid input. Please enter a valid number.")

# Find the closest row in the dataset
closest_row = find_closest_row(np.array(input_values), df)
closest_crop = closest_row['label']
hindi_name = crop_name_mapping.get(closest_crop, "Unknown")

# Print the result
print(f"\nBased on your input, your soil is suitable to grow {closest_crop} ({hindi_name}).")
print(closest_row)

# Ask if user wants crop suggestion
response = input("Do you want to grow a specific crop? (yes/no): ").lower()

if response == "yes":
    # Prompt user to enter the crop name
    crop_name = input("Enter the crop name you want to grow (in English or Hindi): ").strip().lower()

    # Check if the crop name exists in the dataset in either language
    crop_name_english = crop_name_mapping.get(crop_name, None)
    crop_name_hindi = hindi_to_english.get(crop_name, None)

    if crop_name_english or crop_name_hindi:
        selected_crop = crop_name_english if crop_name_english else crop_name_hindi
        crop_data = df[df['label'].str.lower() == selected_crop]

        # Calculate average values for each attribute
        avg_values = crop_data[attributes].mean().values

        # Calculate adjustments needed based on the differences
        adjustments = avg_values - np.array(input_values)

        # Get the Hindi name for the selected crop
        hindi_name = crop_name_mapping.get(selected_crop, "Unknown")

        # Print adjustments needed for each attribute
        print(f"\nTo grow {selected_crop} ({hindi_name}), you need the following adjustments:")
        for attr, adjustment in zip(attributes, adjustments):
            unit = unit_info[attr]
            current_value = input_values[attributes.index(attr)]
            required_value = avg_values[attributes.index(attr)]

            if adjustment > 0:
                print(f"Your {attr} is {current_value:.2f} {unit}, but {selected_crop} ({hindi_name}) requires an average of {required_value:.2f} {unit}.")
                print(f"You need to boost {attr} by {abs(adjustment):.2f} {unit}.")
            elif adjustment < 0:
                print(f"Your {attr} is {current_value:.2f} {unit}, but {selected_crop} ({hindi_name}) requires an average of {required_value:.2f} {unit}.")
                print(f"You have excess {attr} by {abs(adjustment):.2f} {unit}.")
            else:
                print(f"Your {attr} is already optimal for {selected_crop} ({hindi_name}).")
    else:
        print(f"The crop '{crop_name}' is not available in the dataset.")

else:
    print("Thank you for using the program!")