#%% Developed by Deborah, 2024

# Creates a txt with selected attributes from EStreams catalogue based in the gauge_id from excel file

import pandas as pd
import os

# File paths
info_estreams_path = r'D:\My Documents\LoFlowMaas\Discharge\Info_EStreams.xlsx'
soils_file = r'D:\My Documents\LoFlowMaas\EStreams_data\EStreams\attributes\static_attributes\estreams_soil_attributes.csv'
geology_file = r'D:\My Documents\LoFlowMaas\EStreams_data\EStreams\attributes\static_attributes\estreams_geology_attributes.csv'
output_file = r'D:\My Documents\LoFlowMaas\Discharge\all_basins_attributes.txt'

# Load gauge data to get the list of basin_ids
gauge_data = pd.read_excel(info_estreams_path)
basin_ids = gauge_data['basin_id'].dropna().unique()

# Load soils and geology data
soils_data = pd.read_csv(soils_file)
geology_data = pd.read_csv(geology_file)

# Open the output file for writing
with open(output_file, 'w') as file:
    # Write the header
    file.write("Basin_ID\troot_dep\tsoil_tawc\tlit_dom\n")
    
    # Process each basin_id
    for basin_id in basin_ids:
        # Find matching row in soils data
        soil_row = soils_data[soils_data['basin_id'] == basin_id]
        geology_row = geology_data[geology_data['basin_id'] == basin_id]
        
        # Extract root_dep and soil_tawc
        root_dep = soil_row['root_dep_mean'].values[0] if not soil_row.empty else 'NA'
        soil_tawc = soil_row['soil_tawc_mean'].values[0] if not soil_row.empty else 'NA'
        
        # Extract lit_dom
        lit_dom = geology_row['lit_dom'].values[0] if not geology_row.empty else 'NA'
        
        # Write the results to the file
        file.write(f"{basin_id}\t{root_dep}\t{soil_tawc}\t{lit_dom}\n")

print(f"Attributes written to {output_file}")

# Do the same for paired basins as well in the future

#%%