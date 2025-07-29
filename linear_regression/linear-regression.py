from torch import nn, torch
from matplotlib import pyplot as plt

weights = 0.7
biases = 0.3
X = torch.arange(0, 1, 0.02, dtype=torch.float32).unsqueeze(dim=1)
y = weights * X + biases

# create train and test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights: float = nn.Parameter(torch.randn(1),
                                           requires_grad=True,
                                           )
        self.bias: float = nn.Parameter(torch.randn(1),
                                        requires_grad=True,
                                        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))

    # Convert tensors to numpy arrays for plotting
    train_data_np = train_data.detach().numpy() if hasattr(
        train_data, 'detach') else train_data
    train_labels_np = train_labels.detach().numpy() if hasattr(
        train_labels, 'detach') else train_labels
    test_data_np = test_data.detach().numpy() if hasattr(
        test_data, 'detach') else test_data
    test_labels_np = test_labels.detach().numpy() if hasattr(
        test_labels, 'detach') else test_labels

    plt.scatter(train_data_np, train_labels_np, c='b', label='Training data')
    plt.scatter(test_data_np, test_labels_np, c='g', label='Test data')

    if predictions is not None:
        predictions_np = predictions.detach().numpy() if hasattr(
            predictions, 'detach') else predictions
        plt.scatter(test_data_np, predictions_np, c='r', label='Predictions')

    plt.legend()
    plt.show()


torch.manual_seed(42)
model_0 = LinearRegressionModel()

with torch.inference_mode():
    y_pred = model_0(X_train)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

epochs = 200

# Track different values
epoch_count = []
loss_values = []
test_loss_values = []


# Training loop
for epoch in range(epochs):
    # set the model to training mode
    model_0.train()
    # 1 forward pass
    y_pred = model_0(X_train)

    # 2 calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3 optimizer zero grad
    optimizer.zero_grad()

    # 4 perform backpropagation on the loss with respect to the parameters of the models
    loss.backward()

    # 5 step the optimizer (perform gradient descent)
    optimizer.step()

    # set the model to evaluation mode
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f"Epoch: {epoch}, Loss: {loss}, Test Loss: {test_loss}")
        print(model_0.state_dict())

# Save the trained model
model_save_path = "linear_regression_model.pth"
torch.save(model_0.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Load the model (demonstration)


def load_model(model_path):
    """Load a saved model state dict"""
    loaded_model = LinearRegressionModel()
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()  # Set to evaluation mode
    return loaded_model


# Example of loading the model
loaded_model = load_model(model_save_path)
print(f"Model loaded from {model_save_path}")
print(f"Loaded model state: {loaded_model.state_dict()}")

# Make predictions with both original and loaded model to verify they're the same
with torch.inference_mode():
    y_pred_new = model_0(X_test)
    y_pred_loaded = loaded_model(X_test)

    # Verify models produce identical results
    print(
        f"Models produce identical results: {torch.allclose(y_pred_new, y_pred_loaded)}")

# Plot initial predictions (before training) - create a fresh untrained model
torch.manual_seed(42)
initial_model = LinearRegressionModel()
with torch.inference_mode():
    initial_pred = initial_model(X_test)
plot_predictions(predictions=initial_pred)

# Plot final predictions (after training)
plot_predictions(predictions=y_pred_new)

# Plot the loss curves
plt.plot(epoch_count, loss_values, label='Train loss')
plt.plot(epoch_count, test_loss_values, label='Test loss')
plt.title('Training and test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
