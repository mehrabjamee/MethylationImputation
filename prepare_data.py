import polars as pl
import numpy as np
import glob


def prepare_data():
    datasets = []

    # We hold out one dataset for validation, following the paper
    held_out_name = "data/GSE207605_GSE84727.csv"

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
    dnam_all = data.drop(["column", "age"])

    # Process held out validation set similarly
    held_out = held_out.collect().transpose(include_header=True, column_names="")
    age_held_out = held_out.select("age")
    dnam_held_out = held_out.drop(["column", "age"])

    # Left with dnam_filtered, age, dnam_held_out, age_held_out
    print(f"Held out samples:\t{dnam_held_out.shape}\t{age_held_out.shape}")

    # Filter training data to only include CpG sites that are also in validation set
    # This ensures compatibility between training and validation data
    held_out_columns = dnam_held_out.columns
    dnam = dnam_all.select(held_out_columns)

    # Convert polars dataframes to pandas but store in new variables
    dnam_pd = dnam.to_pandas()
    age_pd = age.to_pandas()

    # First split into imputation and GP portions (70/30 split)
    imp_gp_split = np.random.rand(len(dnam_pd)) < 0.7

    # Create imputation portion
    dnam_imp = dnam_pd[imp_gp_split]
    age_imp = age_pd[imp_gp_split]

    # Create GP portion
    dnam_gp = dnam_pd[~imp_gp_split]
    age_gp = age_pd[~imp_gp_split]

    # Split imputation data into train/validation (80/20 split)
    imp_split = np.random.rand(len(dnam_imp)) < 0.8
    dnam_imp_train = dnam_imp[imp_split]
    dnam_imp_val = dnam_imp[~imp_split]
    age_imp_train = age_imp[imp_split]
    age_imp_val = age_imp[~imp_split]

    # Split GP data into train/validation (80/20 split)
    gp_split = np.random.rand(len(dnam_gp)) < 0.8
    dnam_gp_train = dnam_gp[gp_split]
    dnam_gp_val = dnam_gp[~gp_split]
    age_gp_train = age_gp[gp_split]
    age_gp_val = age_gp[~gp_split]

    print(f"Imputation train samples:\t{dnam_imp_train.shape}\t{age_imp_train.shape}")
    print(f"Imputation validation samples:\t{dnam_imp_val.shape}\t{age_imp_val.shape}")
    print(f"GP train samples:\t{dnam_gp_train.shape}\t{age_gp_train.shape}")
    print(f"GP validation samples:\t{dnam_gp_val.shape}\t{age_gp_val.shape}")

    return (
        dnam_held_out,
        age_held_out,
        dnam_imp_train,
        age_imp_train,
        dnam_imp_val,
        age_imp_val,
        dnam_gp_train,
        age_gp_train,
        dnam_gp_val,
        age_gp_val,
    )


if __name__ == "__main__":
    prepare_data()
