import polars as pl
import glob

datasets = []

# We hold out one dataset for validation, following the paper
held_out_name = "data/GSE207605_GSE84727.csv"
held_out_dataset = None

# Load all GSE datasets, separating validation set
for file in glob.glob("data/GSE207605_GSE*.csv"):
    if held_out_name in file:
        held_out = pl.scan_csv(file)
    else:
        datasets.append(pl.scan_csv(file))

# Transpose datasets and combine them into a single dataframe
datasets_t = [
    ds.collect().transpose(include_header=True, column_names="") for ds in datasets
]
data = pl.concat(datasets_t, how="diagonal_relaxed", rechunk=True)

# Split data into age labels and DNA methylation features
age = data.select("age")
dnam = data.drop(["column", "age"])

# Process held out validation set similarly
held_out = held_out.collect().transpose(include_header=True, column_names="")
age_held_out = held_out.select("age")
dnam_held_out = held_out.drop(["column", "age"])

# Get the list of columns from dnam_held_out
held_out_columns = dnam_held_out.columns

# Filter training data to only include CpG sites that are also in validation set
# This ensures compatibility between training and validation data
dnam_filtered = dnam.select(held_out_columns)

# Left with dnam_filtered, age, dnam_held_out, age_held_out
