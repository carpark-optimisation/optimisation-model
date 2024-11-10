import json
import folium
import pandas as pd
from shapely.geometry import shape
from bs4 import BeautifulSoup

# Load the JSON file with the subzone boundaries
with open(r"C:\Users\Newbieshine\Desktop\SMT Project\Master Plan 2019 Subzone Boundary (No Sea) (GEOJSON).geojson", "r") as f:
    data = json.load(f)

# Create a base map centered around Singapore
m = folium.Map(location=[1.3521, 103.8198], zoom_start=12)

# Prepare a list to store subzone data
subzone_data = []

# Process each feature to calculate centroids and add to map
for feature in data["features"]:
    if "geometry" in feature:
        polygon = shape(feature["geometry"])
        centroid = polygon.centroid

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

        # Add the subzone boundary and centroid marker to the map
        folium.GeoJson(
            feature["geometry"],
            style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'weight': 2, 'fillOpacity': 0.2},
            tooltip=subzone_name
        ).add_to(m)

        folium.Marker(
            location=[centroid.y, centroid.x],
            popup=f"Centroid of {subzone_name}",
            tooltip=subzone_name
        ).add_to(m)

        # Append the subzone name and centroid coordinates to subzone_data
        subzone_data.append({
            "Subzone Name": subzone_name,
            "Centroid Latitude": centroid.y,
            "Centroid Longitude": centroid.x
        })

# Save the map as HTML
m.save("subzone_centroids_map.html")
print("Map saved as subzone_centroids_map.html")

# Export subzone data to an Excel file
df = pd.DataFrame(subzone_data)
df.to_excel("subzone_centroids_data.xlsx", index=False)
print("Data exported to subzone_centroids_data.xlsx")

# Load the Excel file with population data
population_df = pd.read_excel(r"C:\Users\Newbieshine\Desktop\SMT Project\SG_Population_Data_2024.xlsx")

# Clean up and prepare the population and subzone data for merging
population_df = population_df[['Subzone', 'Total']].rename(columns={'Subzone': 'Subzone Name', 'Total': 'Total Population'})
population_df['Subzone Name'] = population_df['Subzone Name'].str.lower().str.strip()
subzone_df = pd.DataFrame(subzone_data)
subzone_df['Subzone Name'] = subzone_df['Subzone Name'].str.lower().str.strip()


# # Print unique subzone names for manual comparison
# print("Unique subzone names in the population DataFrame:")
# print(population_df['Subzone Name'].sort_values().unique())
# print("\nUnique subzone names in the subzone DataFrame:")
# print(subzone_df['Subzone Name'].sort_values().unique())
#
# #print the head of the population data and subzone data
# print("Head of the population DataFrame:", population_df.head())
# print("Head of the subzone DataFrame:", subzone_df.head())
#
# print("Column names in the population DataFrame:", population_df.columns)
# print("Column names in the subzone DataFrame:", subzone_df.columns)

# Merge the centroid data with the population data
merged_df = pd.merge(subzone_df, population_df, on='Subzone Name', how='left')

# Export the merged data to a new Excel file or CSV
merged_df.to_excel("merged_single_centroid_subzone_population.xlsx", index=False)
# merged_df.to_csv("merged_subzone_population.csv", index=False)

print("Merged data exported as merged_single_centroid_subzone_population.xlsx and merged_subzone_population.csv")