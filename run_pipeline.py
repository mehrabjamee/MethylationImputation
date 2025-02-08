import pandas as pd
from prepare_data import prepare_data
from train_imputer import train_imputer
from imputation import train_gp_model


def run_imputation_gpr_pipeline(imputation_method):
    (
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
    ) = prepare_data()

    # Train imputer on imputation datasets
    imputer = train_imputer(
        dnam_imp_train,
        age_imp_train,
        dnam_imp_val,
        age_imp_val,
        method=imputation_method,
    )

    # Apply imputation to GP datasets
    imputed_gp_train_dnam, imputed_gp_train_age = imputer(dnam_gp_train, age_gp_train)
    imputed_gp_val_dnam, imputed_gp_val_age = imputer(dnam_gp_val, age_gp_val)

    # Combine imputed train and validation sets
    imputed_train_dnam = pd.concat([imputed_gp_train_dnam, imputed_gp_val_dnam])
    imputed_train_age = pd.concat([imputed_gp_train_age, imputed_gp_val_age])

    # Create training DataFrame with features and age
    train_df = pd.concat([imputed_train_dnam, imputed_train_age], axis=1)

    # Convert polars DataFrames to pandas and create test DataFrame
    dnam_held_out_pd = dnam_held_out.to_pandas()
    age_held_out_pd = age_held_out.to_pandas()
    test_df = pd.concat([dnam_held_out_pd, age_held_out_pd], axis=1)

    # Train GP model
    model, likelihood = train_gp_model(train_df, test_df)

    return model, likelihood


if __name__ == "__main__":
    print("Run evaluation pipeline")
    run_imputation_gpr_pipeline("drop")
    # TODO: Put these debugging prints in the train_imputer function (or better, make tests out of them)
"""
    imputer = train_imputer(
        dnam_imp_train, age_imp_train, dnam_imp_val, age_imp_val, method="vae"
    )

    # Get stats before imputation
    print("\nBefore imputation:")
    print(f"Total rows: {len(dnam_imp_train)}")
    print("dnam")
    print(f"\tRows with missing values: {dnam_imp_train.isna().any(axis=1).sum()}")
    print(f"\tNumber of missing values: {dnam_imp_train.isna().sum().sum()}")
    print("age")
    print(f"\tRows with missing values: {age_imp_train.isna().any(axis=1).sum()}")
    print(f"\tNumber of missing values: {age_imp_train.isna().sum().sum()}")

    # Apply imputation
    imputed_train_dnam, imputed_train_age = imputer(dnam_imp_train, age_imp_train)

    # Get stats after imputation
    print("\nAfter imputation:")
    print(f"Total rows: {len(imputed_train_dnam)}")
    print("dnam")
    print(f"\tRows with missing values: {imputed_train_dnam.isna().any(axis=1).sum()}")
    print(f"\tNumber of missing values: {imputed_train_dnam.isna().sum().sum()}")
    print("age")
    print(f"\tRows with missing values: {imputed_train_age.isna().any(axis=1).sum()}")
    print(f"\tNumber of missing values: {imputed_train_age.isna().sum().sum()}")

    print(f"\nRows removed: {len(dnam_imp_train) - len(imputed_train_dnam)}")
    """
