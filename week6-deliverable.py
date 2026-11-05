"""
IDX Exchange - Week 6
"""

import pandas as pd
import geopandas as gpd
import os
import os
os.environ["SHAPE_RESTORE_SHX"] = "YES"


# ----------------------------------------------------------------------
# Import Datasets
# ----------------------------------------------------------------------
sold = pd.read_csv("sold_cleaned.csv")
districts = gpd.read_file('/Users/belizakbulut/Downloads/DistrictAreas2425')

# ----------------------------------------------------------------------
# Create Key Metrics
# ----------------------------------------------------------------------
sold['Price Ratio'] = sold['ClosePrice'] / sold['OriginalListPrice']
sold['Price Per Sq Ft'] = sold['ClosePrice'] / sold['LivingArea']
sold['Days on Market'] = sold['DaysOnMarket']
sold['Year / Month / YrMo'] = sold['CloseDate']
sold['Close to Original List Ratio'] = sold['ClosePrice'] / sold['OriginalListPrice']
sold['Listing to Contract Days'] = sold['PurchaseContractDate'] - sold['ListingContractDate']
sold['Contract to Close Days'] = sold['CloseDate'] = sold['PurchaseContractDate']

# ----------------------------------------------------------------------
# Create School Districts
# ----------------------------------------------------------------------
props = gpd.GeoDataFrame(
    sold,
    geometry=gpd.points_from_xy(sold["Longitude"], sold["Latitude"]),
    crs="EPSG:4326",
).to_crs(districts.crs)

joined = gpd.sjoin(props, districts, how="left", predicate="within")

sold['School District'] = joined['HighSchoolDistrict']

# ----------------------------------------------------------------------
# Generate Summary Statistics
# ----------------------------------------------------------------------

# PropertyType and PropertySubType
property_type_stats = sold.groupby(['PropertyType','PropertySubType']).describe()

# CountyOrParish and MLSAreaMajor (Commented out for runtime)
# county_area_stats = sold.groupby(['CountyOrParish','MLSAreaMajor']).describe()

# ListOfficeName and BuyerOfficeName (Commented out for runtime)
# competitive_stats = sold.groupby(['ListOfficeName','BuyerOfficeName']).describe()


property_type_stats.head(5)

# ----------------------------------------------------------------------
# Deliverables
# ----------------------------------------------------------------------

new_cols = ['Price Ratio', 'Price Per Sq Ft', 'Days on Market', 
            'Year / Month / YrMo', 'Close to Original List Ratio', 
            'Listing to Contract Days', 'Contract to Close Days', 'School District'
            ]

sold[new_cols].head(5)