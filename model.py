import pandas as pd
import numpy as np
from pulp import *

# Function to calculate Euclidean distance (approximation) between two points (lat, long) in kilometers
def euclidean_distance(lat1, lon1, lat2, lon2):
    lat_km = 111  # Approx. conversion factor for latitude degrees to kilometers
    lon_km = 111 * np.cos(np.radians(lat1))  # Adjust longitude by latitude
    distance = np.sqrt((lat_km * (lat2 - lat1))**2 + (lon_km * (lon2 - lon1))**2)
    return distance

# Subzone data: subzone name, latitude, longitude, population.
# Change this to get the centroids and population data from the previous script.
subzones_data = pd.DataFrame({
    'Subzone': ['Bedok', 'Tampines', 'Ang Mo Kio', 'Yishun', 'Simei'],
    'Latitude': [1.3256, 1.3456, 1.3690, 1.4273, 1.3399],
    'Longitude': [103.9276, 103.9445, 103.8454, 103.8368, 103.9495],
    'Population': [5000, 6000, 7000, 4000, 8000]
})

# Calculate the distance matrix (in kilometers) between each subzone and potential carpark locations
distance_matrix = pd.DataFrame(index=subzones_data['Subzone'], columns=subzones_data['Subzone'])

for i, subzone1 in subzones_data.iterrows():
    for j, subzone2 in subzones_data.iterrows():
        distance_matrix.loc[subzone1['Subzone'], subzone2['Subzone']] = euclidean_distance(
            subzone1['Latitude'], subzone1['Longitude'], subzone2['Latitude'], subzone2['Longitude']
        )

# Maximum allowable distance for coverage (200 meters = 0.2 kilometers)
d_max = 5

# Initialize the optimization problem
prob = LpProblem("Minimize_Carparks", LpMinimize)

# Decision variables for carparks (binary for whether a carpark is placed at each subzone)
X = LpVariable.dicts("Carpark", subzones_data['Subzone'], cat="Binary")

# Decision variables for subzone coverage (binary for whether subzone i is covered by carpark at subzone j)
Y = LpVariable.dicts("Coverage", [(i, j) for i in subzones_data['Subzone'] for j in subzones_data['Subzone']], cat="Binary")

# Objective function: Minimize the number of carparks
prob += lpSum([X[subzone] for subzone in subzones_data['Subzone']])

# Constraints: Each subzone must be covered by at least one carpark
for i in subzones_data['Subzone']:
    prob += lpSum([Y[(i, j)] for j in subzones_data['Subzone']]) >= 1

# Distance constraint: Ensure that subzone i is only covered by subzone j's carpark if distance <= d_max
for i in subzones_data['Subzone']:
    for j in subzones_data['Subzone']:
        if distance_matrix.loc[i, j] > d_max:
            prob += Y[(i, j)] == 0  # Subzone i cannot be covered by carpark j if distance > d_max

# Assignment constraint: Subzones can only be assigned to built carparks
for i in subzones_data['Subzone']:
    for j in subzones_data['Subzone']:
        prob += Y[(i, j)] <= X[j]

# Solve the problem
prob.solve()

# Output the results
print(f"Status: {LpStatus[prob.status]}")

# List of selected carparks and their exact locations (latitude, longitude)
for subzone in subzones_data['Subzone']:
    if X[subzone].varValue == 1:
        lat = subzones_data[subzones_data['Subzone'] == subzone]['Latitude'].values[0]
        lon = subzones_data[subzones_data['Subzone'] == subzone]['Longitude'].values[0]
        print(f"Carpark placed at {subzone} (Latitude: {lat}, Longitude: {lon})")

# Show which subzones are covered by each carpark
for i in subzones_data['Subzone']:
    for j in subzones_data['Subzone']:
        if Y[(i, j)].varValue == 1:
            print(f"Subzone {i} is covered by carpark at {j}")
