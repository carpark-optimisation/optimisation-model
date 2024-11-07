import pandas as pd

# Load the Excel file into a DataFrame
input_file_path = r"C:\Users\Newbieshine\Desktop\SMT Project\Singapore Residents by Planning Area, Subzone, Single Year of Age and Sex, Jun 2024.xlsx"
population_df = pd.read_excel(input_file_path)

# Filter rows where 'Age' and 'Sex' columns are "Total"
filtered_df = population_df[(population_df['Age'] == 'Total') & (population_df['Sex'] == 'Total')]

# Select only the 'Subzone' and '2024' (total population) columns
result_df = filtered_df[['Subzone', 'Total']]

# Save the result to a new Excel file
output_file_path = r"C:\Users\Newbieshine\Desktop\SMT Project\SG_Population_Data_2024.xlsx"
result_df.to_excel(output_file_path, index=False)

print(f"Filtered data exported to {output_file_path}")
