import torch
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp_model(train_df, test_df, n_epochs=200):
    """
    Train a Gaussian Process model to predict age.

    Parameters:
    train_df (pd.DataFrame): Training data with features and 'age' column
    test_df (pd.DataFrame): Test data with same features and 'age' column
    n_epochs (int): Number of training epochs

    Returns:
    tuple: (trained model, likelihood)
    """
    # Validate inputs
    assert "age" in train_df.columns, "'age' column must be present in training data"
    assert "age" in test_df.columns, "'age' column must be present in test data"

    train_features = train_df.drop("age", axis=1).columns
    test_features = test_df.drop("age", axis=1).columns
    assert all(train_features == test_features), (
        "Training and test data must have the same features"
    )

    assert not train_df.isnull().values.any(), "Training data contains missing values"
    assert not test_df.isnull().values.any(), "Test data contains missing values"

    # Prepare data
    X_train = train_df.drop("age", axis=1)
    y_train = train_df["age"]
    X_test = test_df.drop("age", axis=1)
    y_test = test_df["age"]

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values)

    # Initialize model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train_tensor, y_train_tensor, likelihood)

    # Train model
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = -mll(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch {i + 1}/{n_epochs} - Loss: {loss.item():.3f}")

    # Evaluate model
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_pred = likelihood(model(X_test_tensor)).mean
        rmse = torch.sqrt(torch.mean((y_pred - y_test_tensor) ** 2))
        print(f"\nTest RMSE: {rmse.item():.3f} years")

    return model, likelihood
