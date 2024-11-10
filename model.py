from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from pulp import *
from itertools import combinations

app = Flask(__name__)


@app.route('/')
def status():
    return jsonify({"health": "SMT project good v1"}), 200

@app.route('/optimize-carparks-centroids', methods=['POST'])
def optimize_carparks_centroids():
    try:
        # Get d_max from request
        data = request.get_json()
        d_max = float(data.get('d_max', 0.5))  # Default to 0.5 if not provided
        
        # Constants
        max_capacity = 30000000  # Maximum capacity per carpark
        S = 0                 # Minimum distance between carparks (in km)

        # Read subzone data
        subzones_data = pd.read_excel('subzone_centroids_data.xlsx')
        subzones_data.rename(columns={
            'Subzone Name': 'Subzone',
            'Latitude': 'Latitude',
            'Longitude': 'Longitude'
        }, inplace=True)
        subzones_data['Population'] = 1

        # Function to calculate Euclidean distance
        def euclidean_distance(lat1, lon1, lat2, lon2):
            lat_km = 111
            avg_lat = (lat1 + lat2) / 2
            lon_km = 111 * np.cos(np.radians(avg_lat))
            distance = np.sqrt((lat_km * (lat2 - lat1))**2 + (lon_km * (lon2 - lon1))**2)
            return distance

        subzones = subzones_data['Subzone'].tolist()

        # Create distance matrix
        distance_matrix = pd.DataFrame(index=subzones, columns=subzones)
        for i, subzone1 in subzones_data.iterrows():
            for j, subzone2 in subzones_data.iterrows():
                distance = euclidean_distance(
                    subzone1['Latitude'], subzone1['Longitude'],
                    subzone2['Latitude'], subzone2['Longitude']
                )
                distance_matrix.loc[subzone1['Subzone'], subzone2['Subzone']] = distance

        # Create clusters
        clusters = {}
        for subzone_center in subzones:
            subzones_in_cluster = []
            for subzone_target in subzones:
                if float(distance_matrix.loc[subzone_center, subzone_target]) <= d_max:
                    subzones_in_cluster.append(subzone_target)
            clusters[subzone_center] = subzones_in_cluster

        population = subzones_data.set_index('Subzone')['Population'].to_dict()
        latitudes = subzones_data.set_index('Subzone')['Latitude'].to_dict()
        longitudes = subzones_data.set_index('Subzone')['Longitude'].to_dict()

        # Initialize optimization problem
        prob = LpProblem("Set_Cover_Carpark_Problem", LpMinimize)
        M = sum(population.values())
        X = LpVariable.dicts("Cluster", clusters.keys(), cat="Binary")

        # Objective function
        prob += lpSum([X[c] for c in clusters.keys()])

        # Constraints
        for s in subzones:
            prob += lpSum([X[c] for c in clusters.keys() if s in clusters[c]]) >= 1, f"Coverage_{s}"

        for c in clusters.keys():
            total_population = sum([population[s] for s in clusters[c]])
            prob += total_population <= max_capacity + M * (1 - X[c]), f"Capacity_{c}"

        for c1, c2 in combinations(clusters.keys(), 2):
            distance = float(distance_matrix.loc[c1, c2])
            if distance < S:
                prob += X[c1] + X[c2] <= 1, f"Spread_{c1}_{c2}"

        # Solve the problem
        prob.solve()

        # Print statements for checking (retained as requested)
        print(f"Status: {LpStatus[prob.status]}")
        print(f"Optimal number of carparks: {int(value(prob.objective))}\n")

        # Prepare results in the specified format
        results = []
        for c in clusters.keys():
            if X[c].varValue == 1:
                # Calculate centroid
                covered_subzones = clusters[c]
                lat_sum = sum([latitudes[s] for s in covered_subzones])
                lon_sum = sum([longitudes[s] for s in covered_subzones])
                n = len(covered_subzones)
                centroid_lat = lat_sum / n
                centroid_lon = lon_sum / n

                # Print statements for checking (retained as requested)
                print(f"Carpark placed at cluster centered at {c}")
                print(f"  Centroid Latitude: {centroid_lat:.6f}, Longitude: {centroid_lon:.6f}")
                print(f"  Covers subzones: {', '.join(covered_subzones)}")
                print(f"  Total Population Covered: {sum([population[s] for s in covered_subzones])}\n")

                # Add result to list
                results.append({
                    "planning_area_name": c,  # Using cluster center subzone as planning area name
                    "latitude": float(centroid_lat),
                    "longitude": float(centroid_lon),
                    "covers_subzones": covered_subzones
                })

        # Check for uncovered subzones
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

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/optimize-carparks-poisson', methods=['POST'])
def optimize_carparks_poisson():
    try:
        # Get d_max from request
        data = request.get_json()
        d_max = float(data.get('d_max', 0.5))  # Default to 0.5 if not provided
        
        # Constants
        max_capacity = 30000000  # Maximum capacity per carpark
        S = 0                 # Minimum distance between carparks (in km)

        # Read subzone data
        subzones_data = pd.read_excel('subzone_poisson_points_data.xlsx')
        subzones_data.rename(columns={
            'Subzone Name': 'Subzone',
            'Latitude': 'Latitude',
            'Longitude': 'Longitude'
        }, inplace=True)
        subzones_data['Population'] = 1

        # Function to calculate Euclidean distance
        def euclidean_distance(lat1, lon1, lat2, lon2):
            lat_km = 111
            avg_lat = (lat1 + lat2) / 2
            lon_km = 111 * np.cos(np.radians(avg_lat))
            distance = np.sqrt((lat_km * (lat2 - lat1))**2 + (lon_km * (lon2 - lon1))**2)
            return distance

        subzones = subzones_data['Subzone'].tolist()

        # Create distance matrix
        distance_matrix = pd.DataFrame(index=subzones, columns=subzones)
        for i, subzone1 in subzones_data.iterrows():
            for j, subzone2 in subzones_data.iterrows():
                distance = euclidean_distance(
                    subzone1['Latitude'], subzone1['Longitude'],
                    subzone2['Latitude'], subzone2['Longitude']
                )
                distance_matrix.loc[subzone1['Subzone'], subzone2['Subzone']] = distance

        # Create clusters
        clusters = {}
        for subzone_center in subzones:
            subzones_in_cluster = []
            for subzone_target in subzones:
                if float(distance_matrix.loc[subzone_center, subzone_target]) <= d_max:
                    subzones_in_cluster.append(subzone_target)
            clusters[subzone_center] = subzones_in_cluster

        population = subzones_data.set_index('Subzone')['Population'].to_dict()
        latitudes = subzones_data.set_index('Subzone')['Latitude'].to_dict()
        longitudes = subzones_data.set_index('Subzone')['Longitude'].to_dict()

        # Initialize optimization problem
        prob = LpProblem("Set_Cover_Carpark_Problem", LpMinimize)
        M = sum(population.values())
        X = LpVariable.dicts("Cluster", clusters.keys(), cat="Binary")

        # Objective function
        prob += lpSum([X[c] for c in clusters.keys()])

        # Constraints
        for s in subzones:
            prob += lpSum([X[c] for c in clusters.keys() if s in clusters[c]]) >= 1, f"Coverage_{s}"

        for c in clusters.keys():
            total_population = sum([population[s] for s in clusters[c]])
            prob += total_population <= max_capacity + M * (1 - X[c]), f"Capacity_{c}"

        for c1, c2 in combinations(clusters.keys(), 2):
            distance = float(distance_matrix.loc[c1, c2])
            if distance < S:
                prob += X[c1] + X[c2] <= 1, f"Spread_{c1}_{c2}"

        # Solve the problem
        prob.solve()

        # Print statements for checking (retained as requested)
        print(f"Status: {LpStatus[prob.status]}")
        print(f"Optimal number of carparks: {int(value(prob.objective))}\n")

        # Prepare results in the specified format
        results = []
        for c in clusters.keys():
            if X[c].varValue == 1:
                # Calculate centroid
                covered_subzones = clusters[c]
                lat_sum = sum([latitudes[s] for s in covered_subzones])
                lon_sum = sum([longitudes[s] for s in covered_subzones])
                n = len(covered_subzones)
                centroid_lat = lat_sum / n
                centroid_lon = lon_sum / n

                # Print statements for checking (retained as requested)
                print(f"Carpark placed at cluster centered at {c}")
                print(f"  Centroid Latitude: {centroid_lat:.6f}, Longitude: {centroid_lon:.6f}")
                print(f"  Covers subzones: {', '.join(covered_subzones)}")
                print(f"  Total Population Covered: {sum([population[s] for s in covered_subzones])}\n")

                # Add result to list
                results.append({
                    "planning_area_name": c,  # Using cluster center subzone as planning area name
                    "latitude": float(centroid_lat),
                    "longitude": float(centroid_lon),
                    "covers_subzones": covered_subzones
                })

        # Check for uncovered subzones
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

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/optimize-carparks-random', methods=['POST'])
def optimize_carparks_random():
    try:
        # Get d_max from request
        data = request.get_json()
        d_max = float(data.get('d_max', 0.5))  # Default to 0.5 if not provided
        
        # Constants
        max_capacity = 30000000  # Maximum capacity per carpark
        S = 0                 # Minimum distance between carparks (in km)

        # Read subzone data
        subzones_data = pd.read_excel('subzone_random_points_data.xlsx')
        subzones_data.rename(columns={
            'Subzone Name': 'Subzone',
            'Latitude': 'Latitude',
            'Longitude': 'Longitude'
        }, inplace=True)
        subzones_data['Population'] = 1

        # Function to calculate Euclidean distance
        def euclidean_distance(lat1, lon1, lat2, lon2):
            lat_km = 111
            avg_lat = (lat1 + lat2) / 2
            lon_km = 111 * np.cos(np.radians(avg_lat))
            distance = np.sqrt((lat_km * (lat2 - lat1))**2 + (lon_km * (lon2 - lon1))**2)
            return distance

        subzones = subzones_data['Subzone'].tolist()

        # Create distance matrix
        distance_matrix = pd.DataFrame(index=subzones, columns=subzones)
        for i, subzone1 in subzones_data.iterrows():
            for j, subzone2 in subzones_data.iterrows():
                distance = euclidean_distance(
                    subzone1['Latitude'], subzone1['Longitude'],
                    subzone2['Latitude'], subzone2['Longitude']
                )
                distance_matrix.loc[subzone1['Subzone'], subzone2['Subzone']] = distance

        # Create clusters
        clusters = {}
        for subzone_center in subzones:
            subzones_in_cluster = []
            for subzone_target in subzones:
                if float(distance_matrix.loc[subzone_center, subzone_target]) <= d_max:
                    subzones_in_cluster.append(subzone_target)
            clusters[subzone_center] = subzones_in_cluster

        population = subzones_data.set_index('Subzone')['Population'].to_dict()
        latitudes = subzones_data.set_index('Subzone')['Latitude'].to_dict()
        longitudes = subzones_data.set_index('Subzone')['Longitude'].to_dict()

        # Initialize optimization problem
        prob = LpProblem("Set_Cover_Carpark_Problem", LpMinimize)
        M = sum(population.values())
        X = LpVariable.dicts("Cluster", clusters.keys(), cat="Binary")

        # Objective function
        prob += lpSum([X[c] for c in clusters.keys()])

        # Constraints
        for s in subzones:
            prob += lpSum([X[c] for c in clusters.keys() if s in clusters[c]]) >= 1, f"Coverage_{s}"

        for c in clusters.keys():
            total_population = sum([population[s] for s in clusters[c]])
            prob += total_population <= max_capacity + M * (1 - X[c]), f"Capacity_{c}"

        for c1, c2 in combinations(clusters.keys(), 2):
            distance = float(distance_matrix.loc[c1, c2])
            if distance < S:
                prob += X[c1] + X[c2] <= 1, f"Spread_{c1}_{c2}"

        # Solve the problem
        prob.solve()

        # Print statements for checking (retained as requested)
        print(f"Status: {LpStatus[prob.status]}")
        print(f"Optimal number of carparks: {int(value(prob.objective))}\n")

        # Prepare results in the specified format
        results = []
        for c in clusters.keys():
            if X[c].varValue == 1:
                # Calculate centroid
                covered_subzones = clusters[c]
                lat_sum = sum([latitudes[s] for s in covered_subzones])
                lon_sum = sum([longitudes[s] for s in covered_subzones])
                n = len(covered_subzones)
                centroid_lat = lat_sum / n
                centroid_lon = lon_sum / n

                # Print statements for checking (retained as requested)
                print(f"Carpark placed at cluster centered at {c}")
                print(f"  Centroid Latitude: {centroid_lat:.6f}, Longitude: {centroid_lon:.6f}")
                print(f"  Covers subzones: {', '.join(covered_subzones)}")
                print(f"  Total Population Covered: {sum([population[s] for s in covered_subzones])}\n")

                # Add result to list
                results.append({
                    "planning_area_name": c,  # Using cluster center subzone as planning area name
                    "latitude": float(centroid_lat),
                    "longitude": float(centroid_lon),
                    "covers_subzones": covered_subzones
                })

        # Check for uncovered subzones
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

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5061, debug=True)