"""
IDX Exchange - Weeks 4-5: Data Cleaning and Preparation
"""

import warnings
warnings.filterwarnings('ignore', message='Mean of empty slice')

import numpy as np
import pandas as pd

sold = pd.read_csv('sold_enriched.csv')

print("=" * 70)
print("LOAD")
print("=" * 70)
rows_before = len(sold)
print(f"Rows before cleaning: {rows_before:,}")
print(f"Columns before cleaning: {sold.shape[1]}")

# ----------------------------------------------------------------------
# Convert date fields to datetime format
# ----------------------------------------------------------------------
date_cols = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate']
sold[date_cols] = sold[date_cols].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce')

print("\n" + "=" * 70)
print("DATE CONVERSION")
print("=" * 70)
print("Date column types:")
print(sold[date_cols].dtypes.to_string())
print("Null dates after conversion (includes values that could not be parsed):")
print(sold[date_cols].isna().sum().to_string())

# ----------------------------------------------------------------------
# Remove unnecessary / redundant columns
# ----------------------------------------------------------------------
drop_cols = [
    'ListingId', 'ListingKeyNumeric', 'CoveredSpaces', 'AttachedGarageYN',
    'LotSizeAcres', 'LotSizeArea', 'AboveGradeFinishedArea', 'BuildingAreaTotal',
    'Levels', 'FireplaceYN', 'ContractStatusChangeDate', 'year_month',
    'UnparsedAddress', 'StreetNumberNumeric', 'ListAgentEmail',
    'ListAgentFirstName', 'ListAgentLastName', 'CoListAgentFirstName',
    'CoListAgentLastName', 'CoBuyerAgentFirstName', 'BuyerAgentFirstName',
    'BuyerAgentLastName', 'ListAgentAOR', 'BuyerAgentAOR', 'BuyerOfficeAOR',
    'LotSizeDimensions', 'OriginatingSystemName', 'OriginatingSystemSubName',
    'MlsStatus', 'BusinessType', 'MainLevelBedrooms', 'BuyerAgencyCompensation', 'BuyerAgencyCompensationType', 'Flooring',
    'ElementarySchool', 'MiddleOrJuniorSchool', 'HighSchool', 'Latitude', 'Longitude'
]
sold = sold.drop(columns=drop_cols)

print("\n" + "=" * 70)
print("REMOVE REDUNDANT COLUMNS")
print("=" * 70)
print(f"Dropped {len(drop_cols)} columns; {sold.shape[1]} columns remain")

# ----------------------------------------------------------------------
# Handle missing values appropriately
# ----------------------------------------------------------------------

# High missing
high_missing = sold.columns[sold.isna().mean() > 0.90]
sold = sold.drop(columns=high_missing)

# Low missing
missing = sold.isna().mean()
low_missing = missing[(missing > 0) & (missing <= 0.05)].index
rows_pre_dropna = len(sold)
sold = sold.dropna(subset=low_missing)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(f"Columns dropped for >90% missing: {list(high_missing)}")
print(f"Columns with <=5% missing (rows with nulls dropped): {list(low_missing)}") 
print(f"Rows dropped: {rows_pre_dropna - len(sold):,} "
      f"({(rows_pre_dropna - len(sold)) / rows_pre_dropna:.1%}); {len(sold):,} remain")

sold['AssociationFee_was_missing'] = sold['AssociationFee'].isna()
sold['AssociationFee'] = sold['AssociationFee'].fillna(0)
sold['AssociationFeeFrequency'] = sold['AssociationFeeFrequency'].fillna('None')

# Boolean flags: null -> False
for c in ['NewConstructionYN', 'PoolPrivateYN', 'ViewYN']:
    sold[c] = sold[c].fillna(False)

# Indicator instead of the office name
sold['HasCoListAgentOffice'] = sold['CoListOfficeName'].notna()
sold = sold.drop(columns='CoListOfficeName')

# Location text: null -> 'Unknown'
for c in ['HighSchoolDistrict', 'MLSAreaMajor', 'SubdivisionName']:
    sold[c] = sold[c].fillna('Unknown')

print("Structural nulls filled: AssociationFee (0), AssociationFeeFrequency ('None'), "
      "NewConstructionYN/PoolPrivateYN/ViewYN (False), location text ('Unknown')")


def tvd_perm_test(df, col, cat_col, n=200):
    d = pd.DataFrame({'m': df[col].isna().to_numpy(), 'c': df[cat_col].to_numpy()})

    def tvd(d):
        p = d.pivot_table(index='c', columns='m', aggfunc='size').fillna(0)
        p = p / p.sum()
        return p.diff(axis=1).iloc[:, -1].abs().sum() / 2

    obs = tvd(d)
    sims = []
    for _ in range(n):
        d['c'] = np.random.permutation(d['c'])
        sims.append(tvd(d))
    return obs, (np.array(sims) >= obs).mean()


print("\nMissingness permutation tests (does missingness depend on PropertySubType?):")
for c in ['LotSizeSquareFeet', 'Stories']:
    obs, p = tvd_perm_test(sold, c, 'PropertySubType')
    print(f"{c} missingness vs PropertySubType: TVD={obs:.3f}, p={p:.3f}")

# Imputation
for c in ['LotSizeSquareFeet', 'Stories']:
    sold[c + '_was_missing'] = sold[c].isna()

# Group-wise fills, conditional on PropertySubType (what the test found)
sold['LotSizeSquareFeet'] = sold['LotSizeSquareFeet'].fillna(
    sold.groupby('PropertySubType')['LotSizeSquareFeet'].transform('median')
)
sold['Stories'] = sold['Stories'].fillna(
    sold.groupby('PropertySubType')['Stories'].transform(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan
    )
)
# Fallbacks
sold['LotSizeSquareFeet'] = sold['LotSizeSquareFeet'].fillna(0)
sold['Stories'] = sold['Stories'].fillna(sold['Stories'].mode().iloc[0])

rows_pre_coords = len(sold)
for c in ['latfilled', 'lonfilled']:
    sold[c] = sold[c].fillna(sold.groupby('PostalCode')[c].transform('median'))
sold = sold.dropna(subset=['latfilled', 'lonfilled'])

print(f"Imputed LotSizeSquareFeet and Stories within PropertySubType (flags added)")
print(f"Coordinates: rows dropped after postal-code fill: {rows_pre_coords - len(sold):,}")
print(f"Remaining Missing Values: {len((sold.isna().sum()[lambda s: s > 0]))}")

# ----------------------------------------------------------------------
# Ensure numeric fields are properly typed
# ----------------------------------------------------------------------
numeric_fields = [
    'AssociationFee', 'BathroomsTotalInteger', 'BedroomsTotal', 'ClosePrice',
    'DaysOnMarket', 'GarageSpaces', 'latfilled', 'lonfilled',
    'ListPrice', 'OriginalListPrice', 'LivingArea', 'LotSizeSquareFeet',
    'ParkingTotal', 'Stories', 'YearBuilt', 'rate_30yr_fixed',
]

sold[numeric_fields] = sold[numeric_fields].apply(pd.to_numeric, errors='coerce')

sold[['latfilled', 'lonfilled']] = sold[['latfilled', 'lonfilled']].astype('float64')
sold = sold.rename(columns={'latfilled': 'Latitude', 'lonfilled': 'Longitude'})

print("\n" + "=" * 70)
print("NUMERIC TYPE CONFIRMATION")
print("=" * 70)
numeric_final = ['Latitude' if c == 'latfilled' else 'Longitude' if c == 'lonfilled' else c
                 for c in numeric_fields] 
print(sold[numeric_final].dtypes.to_string())  
nan_after_typing = sold[numeric_final].isna().sum()
print(f"Numeric nulls after typing (should be 0): {int(nan_after_typing.sum())}")

# ----------------------------------------------------------------------
# Remove or flag invalid numeric values 
# ----------------------------------------------------------------------
sold['invalid_close_price_flag'] = sold['ClosePrice'] <= 0
sold['invalid_living_area_flag'] = sold['LivingArea'] <= 0
sold['negative_days_on_market_flag'] = sold['DaysOnMarket'] < 0
sold['negative_bed_bath_flag'] = (sold['BedroomsTotal'] < 0) | (sold['BathroomsTotalInteger'] < 0)

numeric_flags = ['invalid_close_price_flag', 'invalid_living_area_flag',
                 'negative_days_on_market_flag', 'negative_bed_bath_flag']
print("\n" + "=" * 70)
print("INVALID NUMERIC VALUES (flagged, rows kept)")
print("=" * 70)
print(sold[numeric_flags].sum().to_string())
print(f"Rows with at least one invalid numeric value: {int(sold[numeric_flags].any(axis=1).sum()):,}")

# ----------------------------------------------------------------------
# Date Consistency
# ----------------------------------------------------------------------
sold['listing_after_close_flag'] = sold['ListingContractDate'] > sold['CloseDate']
sold['purchase_after_close_flag'] = sold['PurchaseContractDate'] > sold['CloseDate']

listing_after_purchase = sold['ListingContractDate'] > sold['PurchaseContractDate'] 
sold['negative_timeline_flag'] = (
    sold['listing_after_close_flag']
    | sold['purchase_after_close_flag']
    | listing_after_purchase
)

listing_flags = ['listing_after_close_flag', 'purchase_after_close_flag', 'negative_timeline_flag']
print("\n" + "=" * 70)
print("DATE CONSISTENCY CHECKS")
print("=" * 70)
print(f"Date Consistency Checks:\n{sold[listing_flags].sum().to_string()}")

# ----------------------------------------------------------------------
# Geographic Data Checks
# ----------------------------------------------------------------------

# Missing Values Check
lat_missing, lon_missing = sold['Latitude'].isna().sum(), sold['Longitude'].isna().sum()

# Invalid Coordinates Flag
sold['missing_coordinates'] = (sold['Latitude'].isna() | sold['Longitude'].isna())
sold['sentinel_coordinates'] = ((sold['Latitude'] == 0) | (sold['Longitude'] == 0))
sold['long_greater_0'] = sold['Longitude'] > 0

in_ca = sold['Latitude'].between(32.5, 42.0) & sold['Longitude'].between(-124.5, -114.1)
sold['out_of_state_coordinates'] = ~in_ca & ~sold['missing_coordinates']

geo_flags = ['missing_coordinates', 'sentinel_coordinates', 'long_greater_0', 'out_of_state_coordinates']

print("\n" + "=" * 70)
print("GEOGRAPHIC DATA QUALITY SUMMARY")
print("=" * 70)
print(f"Number of missing Latitude values: {lat_missing}\nNumber of missing Longitude values: {lon_missing}")
print(f"\nGeographic Data Checks:\n{sold[geo_flags].sum().to_string()}")
print(f"Records with at least one geographic flag: {int(sold[geo_flags].any(axis=1).sum()):,}")

# ----------------------------------------------------------------------
# Final summary and export 
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Rows before cleaning: {rows_before:,}")
print(f"Rows after cleaning : {len(sold):,}  ({rows_before - len(sold):,} removed, "
      f"{(rows_before - len(sold)) / rows_before:.1%})")
print(f"Final shape: {sold.shape[0]:,} rows x {sold.shape[1]} columns")

sold.to_csv('sold_cleaned.csv', index=False)
print("Saved sold_cleaned.csv")
