import pandas as pd
import numpy as np

# Define the crops and environmental threats

crops = ['Coffee', 'Tea', 'Cassava', 'Sweet Potatoes', 'Maize', 'Rice']
environmental_threats = ['Drought', 'Flooding', 'Soil Degradation', 'Pests and Diseases', 'Climate Change']

#Create a list to hold the data
data = []

# Generate random data
for year in range(2015, 2025):  # Years from 2015 to 2024
    for crop in crops:
        for _ in range(100):  # Generate 100 observations for each crop
            area_cultivated = np.random.randint(5, 100)  # Area cultivated between 5 to 100 hectares
            yield_per_hectare = round(np.random.uniform(1.0, 10.0), 2)  # Yield between 1.0 to 10.0 tons per hectare
            threat = np.random.choice(environmental_threats)  # Randomly choose an environmental threat
            
            # Introduce some missing values randomly
            if np.random.rand() < 0.1:  # 10% chance to have a missing value
                area_cultivated = np.nan

            impact_severity = np.random.choice(['Low', 'Medium', 'High'])  # Random impact severity
            food_security_index = round(np.random.uniform(0, 100), 2)  # Food security index between 0 and 100

            # Append the data to the list
            data.append([crop, area_cultivated, yield_per_hectare, threat, impact_severity, food_security_index, year])

# Create a DataFrame
columns = ['Crop', 'Area Cultivated (ha)', 'Yield (tons/ha)', 'Environmental Threat', 
           'Impact Severity', 'Food Security Index', 'Year']
df = pd.DataFrame(data, columns=columns)

# Introduce some missing values randomly in the 'Yield' column
for i in range(len(df)):
    if np.random.rand() < 0.1:  # 10% chance for missing values in Yield
        df.at[i, 'Yield (tons/ha)'] = np.nan


# Save the DataFrame to a CSV file
#df.to_csv('../data/generated_now.csv', index=False)


# Create a list to hold the additional data
additional_data = []

# Define some locations in rural Burundi
locations = ['Gitega', 'Ngozi', 'Muramvya', 'Makamba', 'Bururi', 'Cankuzo']

# Generate additional meaningful data
for year in range(2015, 2025):
    for crop in crops:
        for _ in range(100):  # 100 observations for each crop
            location = np.random.choice(locations)  # Randomly select a location
            soil_quality = np.random.choice(['Good', 'Average', 'Poor'])
            flooding_risk = np.random.choice([0, 1])  # 0: No flooding, 1: Flooding risk
            soil_degradation = np.random.choice([0, 1])  # 0: No degradation, 1: Degradation
            farming_practice = np.random.choice(['Traditional', 'Modern'])

            # Introduce some missing values randomly
            if np.random.rand() < 0.1:  # 10% chance to have a missing value
                soil_quality = np.nan

            # Append the additional data
            additional_data.append([crop, location, year, soil_quality, flooding_risk, soil_degradation, farming_practice])

# Create a DataFrame for additional data
additional_columns = ['Crop', 'Location', 'Year', 'Soil Quality', 'Flooding Risk', 'Soil Degradation', 'Farming Practice']
additional_df = pd.DataFrame(additional_data, columns=additional_columns)


# Optionally, you can merge the two DataFrames (df and additional_df) for combined analysis.
combined_df = pd.merge(df, additional_df, on=['Crop', 'Year'], how='outer')
combined_df.to_csv('burundi_food_security.csv', index=False)

