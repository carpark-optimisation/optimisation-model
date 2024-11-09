import json
import folium
import pandas as pd
from shapely.geometry import shape, Point, MultiPoint
import random
from scipy.spatial import KDTree
import numpy as np
from bs4 import BeautifulSoup

# Function to generate evenly spaced points using Poisson Disk Sampling
def generate_poisson_disk_points(polygon, min_distance=0.001, num_points=5):
    points = []
    attempts = 0
    max_attempts = 10000  # Limit the number of attempts to avoid infinite loops

    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    while len(points) < num_points and attempts < max_attempts:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(random_point):
            # Check if the new point is far enough from existing points
            if all(random_point.distance(p) >= min_distance for p in points):
                points.append(random_point)
        attempts += 1

    return points if len(points) >= num_points else points + [polygon.centroid] * (num_points - len(points))

# Load the JSON file with the subzone boundaries
with open(r"C:\Users\Newbieshine\Desktop\SMT Project\Master Plan 2019 Subzone Boundary (No Sea) (GEOJSON).geojson", "r") as f:
    data = json.load(f)

# Create a base map centered around Singapore
m = folium.Map(location=[1.3521, 103.8198], zoom_start=12)

# Prepare a list to store subzone data
subzone_data = []

# Process each feature to generate multiple coordinates and add to the map
for feature in data["features"]:
    if "geometry" in feature:
        polygon = shape(feature["geometry"])

        # Extract subzone name from Description HTML if properties do not contain SUBZONE_N
        properties = feature.get("properties", {})
        description_html = properties.get("Description", "")
        subzone_name = "Unknown Subzone"

        # Parse the HTML to find SUBZONE_N
        if description_html:
            soup = BeautifulSoup(description_html, "html.parser")
            subzone_n_td = soup.find('th', string="SUBZONE_N")

            if subzone_n_td:
                subzone_name = subzone_n_td.find_next_sibling("td").text.strip()

        # Add the subzone boundary to the map
        folium.GeoJson(
            feature["geometry"],
            style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'weight': 1, 'fillOpacity': 0.2},
            tooltip=subzone_name
        ).add_to(m)

        # Generate Poisson disk points within the subzone
        points = generate_poisson_disk_points(polygon, min_distance=0.005, num_points=5)

        for i, point in enumerate(points):
            # Add each point to the map and subzone data
            point_name = f"{subzone_name} {i+1}"
            folium.Marker(
                location=[point.y, point.x],
                popup=f"Point in {point_name}",
                tooltip=point_name
            ).add_to(m)

            subzone_data.append({
                "Subzone Name": point_name,
                "Main Subzone Name": subzone_name,  # Add main subzone name for merging
                "Latitude": point.y,
                "Longitude": point.x
            })

# Save the map as HTML
m.save("subzone_poisson_points_map.html")
print("Map saved as subzone_poisson_points_map.html")

# Export subzone data to an Excel file
df = pd.DataFrame(subzone_data)
df.to_excel("subzone_poisson_points_data.xlsx", index=False)
print("Data exported to subzone_poisson_points_data.xlsx")

# Load the Excel file with population data
population_df = pd.read_excel(r"C:\Users\Newbieshine\Desktop\SMT Project\SG_Population_Data_2024.xlsx")

# Clean up and prepare the population and subzone data for merging
population_df = population_df[['Subzone', 'Total']].rename(columns={'Subzone': 'Main Subzone Name', 'Total': 'Total Population'})
population_df['Main Subzone Name'] = population_df['Main Subzone Name'].str.lower().str.strip()
subzone_df = pd.DataFrame(subzone_data)
subzone_df['Main Subzone Name'] = subzone_df['Main Subzone Name'].str.lower().str.strip()

# Merge the subzone data with the population data using the main subzone name
merged_df = pd.merge(subzone_df, population_df, on='Main Subzone Name', how='left')

# Export the merged data to a new Excel file or CSV
merged_df.to_excel("merged_poisson_subzone_population.xlsx", index=False)
# merged_df.to_csv("merged_poisson_subzone_population.csv", index=False)

print("Merged data exported as merged_poisson_subzone_population.xlsx and merged_poisson_subzone_population.csv")
