import pandas as pd
import numpy as np
from pulp import *
from itertools import combinations

# Subzone data: subzone name, latitude, longitude, population.
# # Hardcode for test
# subzones_data = pd.DataFrame({
#     'Subzone': ['Bedok', 'Tampines', 'Ang Mo Kio', 'Yishun', 'Simei'],
#     'Latitude': [1.3256, 1.3456, 1.3690, 1.4273, 1.3399],
#     'Longitude': [103.9276, 103.9445, 103.8454, 103.8368, 103.9495],
#     'Population': [5000, 6000, 7000, 4000, 8000]
# })

# Read subzone data from one of the three files
subzones_data = pd.read_excel('Starting_point_generation_codes/starting_points_results/subzone_centroids_data.xlsx')
# subzones_data = pd.read_excel('subzone_poisson_points_data.xlsx')
# subzones_data = pd.read_excel('subzone_random_points_data.xlsx')
subzones_data.rename(columns={
    'Subzone Name': 'Subzone',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude'
}, inplace=True)
subzones_data['Population'] = 1



max_capacity = 30000000  # Maximum capacity per carpark
d_max = 10            # Maximum allowable distance for coverage (in km)
S = 0              # Minimum distance between carparks (in km)

# Function to calculate Euclidean distance (approximation) between two points (lat, lon) in kilometers
def euclidean_distance(lat1, lon1, lat2, lon2):
    lat_km = 111  # Approximate conversion factor for latitude degrees to kilometers
    avg_lat = (lat1 + lat2) / 2
    lon_km = 111 * np.cos(np.radians(avg_lat))  # Adjust longitude by average latitude
    distance = np.sqrt((lat_km * (lat2 - lat1))**2 + (lon_km * (lon2 - lon1))**2)
    return distance

subzones = subzones_data['Subzone'].tolist()

# Create a DataFrame to hold distances between subzones
distance_matrix = pd.DataFrame(index=subzones, columns=subzones)

for i, subzone1 in subzones_data.iterrows():
    for j, subzone2 in subzones_data.iterrows():
        distance = euclidean_distance(
            subzone1['Latitude'], subzone1['Longitude'],
            subzone2['Latitude'], subzone2['Longitude']
        )
        distance_matrix.loc[subzone1['Subzone'], subzone2['Subzone']] = distance

# Create clusters: For each subzone, create a cluster including subzones within d_max
clusters = {}
for subzone_center in subzones:
    # Find subzones within d_max of this subzone_center
    subzones_in_cluster = []
    for subzone_target in subzones:
        if float(distance_matrix.loc[subzone_center, subzone_target]) <= d_max:
            subzones_in_cluster.append(subzone_target)
    clusters[subzone_center] = subzones_in_cluster

population = subzones_data.set_index('Subzone')['Population'].to_dict()
latitudes = subzones_data.set_index('Subzone')['Latitude'].to_dict()
longitudes = subzones_data.set_index('Subzone')['Longitude'].to_dict()

# Big M for capacity constraint (set to total population)
M = sum(population.values())

# Initialize the optimization problem
prob = LpProblem("Set_Cover_Carpark_Problem", LpMinimize)

# Decision variables: Binary variables indicating if a cluster (carpark) is selected
X = LpVariable.dicts("Cluster", clusters.keys(), cat="Binary")

# Objective function: Minimize the number of clusters selected
prob += lpSum([X[c] for c in clusters.keys()])

# Constraints:

# 1. Coverage Constraint: Each subzone must be covered by at least one cluster
for s in subzones:
    prob += lpSum([X[c] for c in clusters.keys() if s in clusters[c]]) >= 1, f"Coverage_{s}"

# 2. Capacity Constraint: Total population in a cluster cannot exceed max_capacity when selected
for c in clusters.keys():
    total_population = sum([population[s] for s in clusters[c]])
    # Use Big M method to relax the constraint when the cluster is not selected
    prob += total_population <= max_capacity + M * (1 - X[c]), f"Capacity_{c}"

# 3. Spread Constraint: Carparks (clusters) must be at least S km apart
for c1, c2 in combinations(clusters.keys(), 2):
    distance = float(distance_matrix.loc[c1, c2])
    if distance < S:
        prob += X[c1] + X[c2] <= 1, f"Spread_{c1}_{c2}"

# Solve the problem
prob.solve()

# Output the results
print(f"Status: {LpStatus[prob.status]}")
print(f"Optimal number of carparks: {int(value(prob.objective))}\n")

# List of selected clusters and their locations
for c in clusters.keys():
    if X[c].varValue == 1:
        # Calculate centroid of the cluster's subzones for carpark location
        lat_sum = sum([latitudes[s] for s in clusters[c]])
        lon_sum = sum([longitudes[s] for s in clusters[c]])
        n = len(clusters[c])
        centroid_lat = lat_sum / n
        centroid_lon = lon_sum / n
        print(f"Carpark placed at cluster centered at {c}")
        print(f"  Centroid Latitude: {centroid_lat:.6f}, Longitude: {centroid_lon:.6f}")
        print(f"  Covers subzones: {', '.join(clusters[c])}")
        print(f"  Total Population Covered: {sum([population[s] for s in clusters[c]])}\n")


uncovered_subzones = []
for s in subzones:
    is_covered = any(s in clusters[c] for c in clusters.keys())
    if not is_covered:
        uncovered_subzones.append(s)

if uncovered_subzones:
    print("The following subzones are not included in any cluster:")
    for s in uncovered_subzones:
        print(s)
    print("Adjust 'd_max' or check subzone coordinates.")
else:
    print("All subzones are included in at least one cluster.")

# Output the results
print(f"Status: {LpStatus[prob.status]}")
print(f"Optimal number of carparks: {int(value(prob.objective))}\n")