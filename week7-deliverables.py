import pandas as pd
sold = pd.read_csv('sold_cleaned.csv')

# ----------------------------------------------------------------------
# Define IQR
# ----------------------------------------------------------------------
def iqr_flag(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)

# ----------------------------------------------------------------------
# Apply to Key Metrics
# ----------------------------------------------------------------------
cols = ['ClosePrice', 'LivingArea', 'DaysOnMarket']
flags = sold[cols].apply(iqr_flag).add_suffix('_outlier')
sold = sold.join(flags)
sold['any_outlier'] = flags.any(axis=1)
no_outlier = sold[~sold['any_outlier']]

# ----------------------------------------------------------------------
# Export New Datasets
# ----------------------------------------------------------------------
sold.to_csv('sold_flagged.csv', index=False)
no_outlier.to_csv('no_outlier.csv', index=False)

# ----------------------------------------------------------------------
# Before and After Comparison
# ----------------------------------------------------------------------
comparison = pd.DataFrame({
    'median_before': sold[cols].median(),
    'median_after': no_outlier[cols].median(),
})
comparison['pct_change'] = (comparison['median_after'] / comparison['median_before'] - 1) * 100

print(f"Rows before: {len(sold):,} | after: {len(no_outlier):,} | "
      f"removed: {len(sold) - len(no_outlier):,} ({sold['any_outlier'].mean():.1%})")
print(comparison.round(2))
print(flags.sum())