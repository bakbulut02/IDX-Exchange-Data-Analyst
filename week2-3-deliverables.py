import numpy as np
import pandas as pd
import plotly.express as px

# ======================================================================
# LOAD DATA
# ======================================================================
l_unfiltered = pd.read_csv("listings.csv")
s_unfiltered = pd.read_csv("sold.csv")
pd.set_option('display.max_columns', None)


# ======================================================================
# MISSING VALUES
# ======================================================================

# Identify high-missing columns
print(l_unfiltered.isna().sum().sort_values(ascending= False)[:10])

# Separate market analysis fields from metadata fields
market_analysis_cols = {
    'AboveGradeFinishedArea': 'market_analysis',
    'AssociationFee': 'market_analysis',
    'AssociationFeeFrequency': 'market_analysis',
    'AttachedGarageYN': 'market_analysis',
    'BathroomsTotalInteger': 'market_analysis',
    'BedroomsTotal': 'market_analysis',
    'BelowGradeFinishedArea': 'market_analysis',
    'BuilderName': 'market_analysis',
    'BuildingAreaTotal': 'market_analysis',
    'BusinessType': 'market_analysis',
    'BuyerAgencyCompensation': 'market_analysis',
    'BuyerAgencyCompensationType': 'market_analysis',
    'City': 'market_analysis',
    'CloseDate': 'market_analysis',
    'ClosePrice': 'market_analysis',
    'ContractStatusChangeDate': 'market_analysis',
    'CountyOrParish': 'market_analysis',
    'CoveredSpaces': 'market_analysis',
    'DaysOnMarket': 'market_analysis',
    'ElementarySchool': 'market_analysis',
    'ElementarySchoolDistrict': 'market_analysis',
    'FireplaceYN': 'market_analysis',
    'FireplacesTotal': 'market_analysis',
    'GarageSpaces': 'market_analysis',
    'HighSchool': 'market_analysis',
    'HighSchoolDistrict': 'market_analysis',
    'Latitude': 'market_analysis',
    'Levels': 'market_analysis'
}

# Calculate missing counts and percentages per column
null_stats = l_unfiltered.isna().agg(['sum', 'mean']).T
null_stats.columns = ['num_null', 'p_null']
print(null_stats.sort_values(by='p_null', ascending=False))

# Flag colunmns with > 90% missing values
print(null_stats[null_stats['p_null'] > 0.9].sort_values(by='p_null', ascending= False))

# Decide which columns to drop vs. retain
drop_cols = [
    'AboveGradeFinishedArea',          # 100% 
    'ElementarySchoolDistrict',        # 100% 
    'TaxYear',                         # 100% 
    'TaxAnnualAmount',                 # 100% 
    'BusinessType',                    # 100%
    'CoveredSpaces',                   # 100%
    'FireplacesTotal',                 # 100%
    'MiddleOrJuniorSchoolDistrict',    # 100%

    'BelowGradeFinishedArea',          # 99.4%
    'CoBuyerAgentFirstName',           # 97.5%
    'BuilderName',                     # 95.4%
    'LotSizeDimensions',               # 94.8%
    'BuildingAreaTotal',               # 91.2%

    'ElementarySchool',                # 88.0%
    'MiddleOrJuniorSchool',            # 88.0%
    'BuyerAgencyCompensation',         # 85.6%
    'BuyerAgencyCompensationType',     # 85.6%
    'HighSchool',                      # 84.3%

    'CoListAgentFirstName',            # 77.7%
    'CoListAgentLastName',             # 77.7%
    'CoListOfficeName',                # 77.6%

    'BuyerOfficeAOR',                  # 74.3%
    'BuyerOfficeName.1',               # 72.6%
    'BuyerOfficeName',                 # 72.6%
    'BuyerAgentFirstName',             # 72.3%
    'BuyerAgentMlsId',                 # 72.2%
    'BuyerAgentLastName',              # 72.2%
]

metadata_to_drop = [
    'ListAgentEmail',
    'ListAgentFirstName',
    'ListAgentFirstName.1',
    'ListAgentFullName',
    'ListAgentLastName',
    'ListAgentLastName.1',
    'ListOfficeName'
]


# ======================================================================
# NUMERIC DISTRIBUTION REVIEW
# ======================================================================
pd.set_option('display.max_rows', 20)
key_num_fields = [
    'ClosePrice',
    'ListPrice',
    'OriginalListPrice',
    'LivingArea',
    'LotSizeAcres',
    'BedroomsTotal',
    'BathroomsTotalInteger',
    'DaysOnMarket',
    'YearBuilt'
    ]
print(s_unfiltered[key_num_fields].describe())


# ======================================================================
# EDA FUNCTION
# ======================================================================
def run_eda(df, name):
    print(f"\n{'=' * 60}\n{name.upper()} EDA\n{'=' * 60}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    # Missing counts and percentages per column
    null_stats = df.isna().agg(["sum", "mean"]).T
    null_stats.columns = ["num_null", "pct_null"]
    null_stats["pct_null"] = (null_stats["pct_null"] * 100).round(2)
    null_stats = null_stats.sort_values("pct_null", ascending=False)
    print(f"NULL STATS:\n{null_stats.to_string()}\n")

    # Flag columns with > 90% missing
    high_missing = null_stats[null_stats["pct_null"] > 90]
    print(f"COLUMNS WITH >90% MISSING ({len(high_missing)}):")
    print(high_missing["pct_null"].to_string())

    return null_stats


run_eda(s_unfiltered, 'sold')

print(s_unfiltered)


# ======================================================================
# VISUALIZATIONS
# ======================================================================
BLUE = "#4C78A8"

# Pricing Vizualizations

price_ticks = [50e3, 100e3, 250e3, 500e3, 1e6, 2.5e6, 5e6]
price_labels = ["$50K", "$100K", "$250K", "$500K", "$1M", "$2.5M", "$5M"]

for col in ["ClosePrice", "ListPrice", "OriginalListPrice"]:
    s = s_unfiltered[col].where(s_unfiltered[col] > 0).dropna()
    lo, hi = s.quantile([0.01, 0.99])
    s = s[(s >= lo) & (s <= hi)]                      # zoom to p1-p99

    fig = px.histogram(pd.DataFrame({col: np.log10(s)}), x=col, nbins=50,
                       marginal="box", color_discrete_sequence=[BLUE],
                       template="plotly_white")
    keep = [(t, l) for t, l in zip(price_ticks, price_labels) if lo <= t <= hi]
    fig.update_xaxes(tickvals=[np.log10(t) for t, _ in keep],
                     ticktext=[l for _, l in keep], title=col + " (log scale)")
    fig.update_yaxes(title_text="Number of listings", row=1, col=1)
    fig.update_layout(title=col, bargap=0.05, showlegend=False)
    fig.update_layout(yaxis_domain=[0, 0.65], yaxis2_domain=[0.75, 1])
    fig.show()


# LivingArea Visualizations
col = "LivingArea"
s = s_unfiltered[col].where(s_unfiltered[col] > 0).dropna()
lo, hi = s.quantile([0.01, 0.99])
s = s[(s >= lo) & (s <= hi)]

fig = px.histogram(s.to_frame(), x=col, nbins=50, marginal="box",
                   color_discrete_sequence=[BLUE], template="plotly_white")
fig.update_xaxes(title="Living area (sq ft)")
fig.update_yaxes(title_text="Number of listings", row=1, col=1)
fig.update_layout(title=col, bargap=0.05, showlegend=False)
fig.update_layout(yaxis_domain=[0, 0.65], yaxis2_domain=[0.75, 1])
fig.show()


# LotSizeAcres Visualization
col = "LotSizeAcres"
s = s_unfiltered[col].where(s_unfiltered[col] > 0).dropna()
lo, hi = s.quantile([0.01, 0.99])
s = s[(s >= lo) & (s <= hi)]

lot_ticks = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
keep = [t for t in lot_ticks if lo <= t <= hi]

fig = px.histogram(pd.DataFrame({col: np.log10(s)}), x=col, nbins=50,
                   marginal="box", color_discrete_sequence=[BLUE],
                   template="plotly_white")
fig.update_xaxes(tickvals=[np.log10(t) for t in keep],
                 ticktext=[f"{t:g}" for t in keep], title="Acres (log scale)")
fig.update_yaxes(title_text="Number of listings", row=1, col=1)
fig.update_layout(title=col + " (zeros excluded)", bargap=0.05, showlegend=False)
fig.update_layout(yaxis_domain=[0, 0.65], yaxis2_domain=[0.75, 1])
fig.show()


# BedroomsTotal and Bathrooms Total Visualizations
for col in ["BedroomsTotal", "BathroomsTotalInteger"]:
    v = s_unfiltered[col].dropna().clip(upper=10).astype(int)
    counts = v.value_counts().sort_index()

    fig = px.bar(x=counts.index.astype(str), y=counts.values,
                 color_discrete_sequence=[BLUE], template="plotly_white",
                 labels={"x": col + "  (10 = 10 or more)", "y": "Number of listings"})
    fig.update_layout(title=col)
    fig.show()


# DaysOnMarket Visualization
col = "DaysOnMarket"
s = s_unfiltered[col].dropna()
lo, hi = s.quantile([0.01, 0.99])
s = s[(s >= lo) & (s <= hi)]

days_fig = px.histogram(s.to_frame(), x=col, nbins=50, marginal="box",
                   color_discrete_sequence=[BLUE], template="plotly_white")
days_fig.update_xaxes(title="Days on market")
days_fig.update_yaxes(title_text="Number of listings", row=1, col=1)
days_fig.update_layout(title=col, bargap=0.05, showlegend=False)
days_fig.show()


# YearBuilt Visualization
col = "YearBuilt"
s = s_unfiltered[col].dropna()
lo, hi = s.quantile([0.01, 0.99])
s = s[(s >= lo) & (s <= hi)]

fig = px.histogram(s.to_frame(), x=col, marginal="box",
                   color_discrete_sequence=[BLUE], template="plotly_white")
fig.update_traces(xbins=dict(size=5), selector=dict(type="histogram"))   # 5-year bins
fig.update_xaxes(title="Year built")
fig.update_yaxes(title_text="Number of listings", row=1, col=1)
fig.update_layout(title=col, bargap=0.05, showlegend=False)
fig.show()


# ======================================================================
# SUGGESTED EDA QUESTIONS
# ======================================================================

# What is the Residential vs. other property type share?
p_resident = (l_unfiltered['PropertyType'] == 'Residential').mean() * 100
print(f"{p_resident}% of all listings are residential.")

# What are the median and average close prices?
close_median  = s_unfiltered['ClosePrice'].median().round(2)
close_avg = s_unfiltered['ClosePrice'].mean().round(2)
print(f"The median close price is ${close_median} and the average close price is ${close_avg}.")

# What does the Days on Market distribution look like?
#days_fig.show()

# What percentage of homes sold above vs. below list price?
p_above = ((s_unfiltered['ClosePrice'] > s_unfiltered['ListPrice']).mean() * 100).round(2)
print(f"{p_above}% of homes sold above the list price.")


# Are there any apparent date consistency issues (e.g., close date before listing date)?
date_issue = (s_unfiltered['ListingContractDate'] > s_unfiltered['CloseDate']).sum()
negative_date = (s_unfiltered['DaysOnMarket'] < 0).sum()
print(f"There are {date_issue} rows with close dates before the listing dates and {negative_date} rows with negative DaysOnMarket.")

# Which counties have the highest median prices?

county_highest = s_unfiltered.groupby('CountyOrParish')['ClosePrice'].median().sort_values(ascending=False)[0:9]
print(f"The top 10 counties with the highest median prices are:\n\n{county_highest}")
