import pandas as pd
import glob

# Read in CSV files
all_listed_dfs = [pd.read_csv(one_file) 
                  for one_file in glob.glob('csv/CRMLSListing*.csv')
                ]
all_sold_dfs = [pd.read_csv(one_file)
                for one_file in glob.glob('csv/CRMLSSold*.csv')
                ]
# Concatenate monthly dataframes
listings = pd.concat(all_listed_dfs, sort=True, ignore_index=True) # 860,898 rows before filter
sold = pd.concat(all_sold_dfs, sort=True, ignore_index=True) # 615,707 rows before filter

# Filter listings and sold dataframes
listings_filtered = listings[listings['PropertyType'] == 'Residential'] # 547,162 rows after filter
sold_filtered = sold[sold['PropertyType'] == 'Residential'] # 414,054 rows after filter


# Export listings and sold dataframes to CSV
listings_filtered.to_csv('listings.csv', index=False)
sold_filtered.to_csv('sold.csv', index=False)
