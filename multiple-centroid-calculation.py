import json
import folium
import pandas as pd
from shapely.geometry import shape, Point
import numpy as np
from bs4 import BeautifulSoup

from scipy.spatial import distance_matrix

# Load the JSON file with the subzone boundaries
with open(r"C:\Users\Newbieshine\Desktop\SMT Project\Master Plan 2019 Subzone Boundary (No Sea) (GEOJSON).geojson", "r") as f:
    data = json.load(f)

# Create a base map centered around Singapore
m = folium.Map(location=[1.3521, 103.8198], zoom_start=12)

# Prepare a list to store subzone data
subzone_data = []

# Function to generate exactly 5 evenly spaced points within a polygon
def generate_exact_points_within_polygon(polygon, num_points=5):
    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Create a grid of points within the bounding box
    x_coords = np.linspace(minx, maxx, 100)  # Increase number for a finer grid
    y_coords = np.linspace(miny, maxy, 100)
    points = [Point(x, y) for x in x_coords for y in y_coords if polygon.contains(Point(x, y))]

    # If fewer points are found than required, fill with centroids
    if len(points) < num_points:
        return points + [polygon.centroid] * (num_points - len(points))

    # Select exactly num_points from the generated points for even distribution
    points_array = np.array([[p.x, p.y] for p in points])
    initial_points = [points_array[0]]  # Start with the first point

    # Greedily select the next farthest point
    for _ in range(1, num_points):
        distances = distance_matrix(initial_points, points_array)
        farthest_index = np.argmax(np.min(distances, axis=0))
        initial_points.append(points_array[farthest_index])

    # Convert back to Point objects
    selected_points = [Point(p[0], p[1]) for p in initial_points]

    return selected_points

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

        # Generate points within the subzone
        points = generate_exact_points_within_polygon(polygon, num_points=5)

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
                "Latitude": point.y,
                "Longitude": point.x
            })

# Save the map as HTML
m.save("subzone_multiple_points_map.html")
print("Map saved as subzone_points_map.html")

# Export subzone data to an Excel file
df = pd.DataFrame(subzone_data)
df.to_excel("subzone_multiple_points_data.xlsx", index=False)
print("Data exported to subzone_multiple_points_data.xlsx")
