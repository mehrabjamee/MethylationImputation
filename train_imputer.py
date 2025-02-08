import pandas as pd
from sklearn.impute import KNNImputer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def train_imputer(train_dnam, train_age, val_dnam, val_age, method="drop"):
    """
    Train an imputer on training data and tune on validation data.

    Args:
        train_dnam (pd.DataFrame): Training DNA methylation data with possibly missing values
        train_age (pd.DataFrame): Training age data corresponding to the DNA methylation data
        val_dnam (pd.DataFrame): Validation DNA methylation data with possibly missing values
        val_age (pd.DataFrame): Validation age data corresponding to the DNA methylation data
        method (str): Imputation method to use. One of ['drop', 'mean', 'knn', 'vae']

    Returns:
        callable: An imputer function that takes (dnam_df, age_df) and returns (imputed_dnam_df, imputed_age_df)
    """

    if method not in ["drop", "mean", "knn", "vae"]:
        raise ValueError(
            f"Method {method} not recognized. Must be one of ['drop', 'mean', 'knn', 'vae']"
        )

    if method == "drop":

        def drop_imputer(dnam_df, age_df):
            mask = ~dnam_df.isna().any(axis=1) & ~age_df.isna().any(axis=1)
            return dnam_df[mask].copy(), age_df[mask].copy()

        return drop_imputer

    elif method == "mean":
        # Calculate mean values for each feature from training data
        feature_means = train_dnam.mean()
        age_mean = train_age.mean()

        def mean_imputer(dnam_df, age_df):
            # Make copies to avoid modifying original data
            imputed_dnam = dnam_df.copy()
            imputed_age = age_df.copy()

            # Fill missing values with means from training data
            imputed_dnam.fillna(feature_means, inplace=True)
            imputed_age.fillna(age_mean, inplace=True)

            return imputed_dnam, imputed_age

        return mean_imputer

    elif method == "knn":
        # Try different k values and evaluate on validation set
        best_score = float("inf")
        k_values = [3, 5, 7, 10, 15]

        # Combine DNA methylation and age data for KNN imputation
        train_combined = pd.concat([train_dnam, train_age], axis=1)
        val_combined = pd.concat([val_dnam, val_age], axis=1)

        for k in k_values:
            # Initialize and fit KNN imputer
            imputer = KNNImputer(n_neighbors=k)
            imputer.fit(train_combined)

            # Impute validation data
            val_imputed = pd.DataFrame(
                imputer.transform(val_combined),
                columns=val_combined.columns,
                index=val_combined.index,
            )

            # Calculate MSE on known values in validation set
            val_mask = ~val_combined.isna()
            mse = ((val_imputed[val_mask] - val_combined[val_mask]) ** 2).mean().mean()

            if mse < best_score:
                best_score = mse
                best_imputer = imputer

        def knn_imputer(dnam_df, age_df):
            # Combine data for imputation
            combined = pd.concat([dnam_df, age_df], axis=1)

            # Apply imputation
            imputed = pd.DataFrame(
                best_imputer.transform(combined),
                columns=combined.columns,
                index=combined.index,
            )

            # Split back into dnam and age
            imputed_dnam = imputed.iloc[:, : dnam_df.shape[1]]
            imputed_age = imputed.iloc[:, dnam_df.shape[1] :]

            return imputed_dnam, imputed_age

        return knn_imputer

    elif method == "vae":
        # FIX: This doesn't impute values properly! This leaves missing rows for some reason, but doesn't even leave them the same, exactly. Look into this before proceeding with more advanced generative models.

        # Get complete data by dropping rows with missing values
        train_mask = ~train_dnam.isna().any(axis=1) & ~train_age.isna().any(axis=1)
        complete_train_dnam = train_dnam[train_mask].copy()
        complete_train_age = train_age[train_mask].copy()

        # Combine DNA methylation and age data
        complete_train_data = pd.concat(
            [complete_train_dnam, complete_train_age], axis=1
        )
        input_dim = complete_train_data.shape[1]

        # Define VAE architecture
        class VAE(nn.Module):
            def __init__(
                self,
                input_dim,
                hidden_dim=16,
                latent_dim=4,
                noise_level=0.1,
                num_epochs=50,
            ):
                super(VAE, self).__init__()
                self.noise_level = noise_level
                self.num_epochs = num_epochs

                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.fc_mean = nn.Linear(hidden_dim, latent_dim)  # Latent mean
                self.fc_log_var = nn.Linear(
                    hidden_dim, latent_dim
                )  # Latent log variance

                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )

            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mean(h), self.fc_log_var(h)

            def reparameterize(self, mu, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std

            def forward(self, x):
                mu, log_var = self.encode(x)
                z = self.reparameterize(mu, log_var)
                return self.decoder(z), mu, log_var

        # Train VAE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae = VAE(input_dim).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)

        def mask_input(x, mask_ratio=0.1):
            mask = torch.rand_like(x) > mask_ratio
            return x * mask

        # Convert data to tensor
        train_tensor = torch.FloatTensor(complete_train_data.values).to(device)

        # Training loop
        vae.train()
        n_epochs = 100
        batch_size = 128

        for epoch in range(n_epochs):
            for i in range(0, len(train_tensor), batch_size):
                batch = train_tensor[i : i + batch_size]
                masked_batch = mask_input(batch)

                # Forward pass
                recon_batch, mu, log_var = vae(masked_batch)

                # Loss calculation
                recon_loss = F.mse_loss(recon_batch, batch)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + 0.1 * kl_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        def vae_imputer(dnam_df, age_df):
            # Combine data
            combined = pd.concat([dnam_df, age_df], axis=1)

            # Convert to tensor
            data_tensor = torch.FloatTensor(combined.values).to(device)

            # Get imputed values
            vae.eval()
            with torch.no_grad():
                imputed_tensor, _, _ = vae(data_tensor)

            # Convert back to DataFrame
            imputed = pd.DataFrame(
                imputed_tensor.cpu().numpy(),
                columns=combined.columns,
                index=combined.index,
            )

            # Split back into dnam and age
            imputed_dnam = imputed.iloc[:, : dnam_df.shape[1]]
            imputed_age = imputed.iloc[:, dnam_df.shape[1] :]

            return imputed_dnam, imputed_age

        return vae_imputer
