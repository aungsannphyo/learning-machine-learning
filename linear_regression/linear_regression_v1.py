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


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


torch.manual_seed(42)
model_1 = LinearRegressionModel()

with torch.inference_mode():
    y_pred = model_1(X_train)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

epochs = 200

# Track different values
epoch_count = []
loss_values = []
test_loss_values = []

# Training loop
for epoch in range(epochs):
    model_1.train()
    y_pred = model_1(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f"Epoch: {epoch}, Loss: {loss}, Test Loss: {test_loss}")
        print(model_1.state_dict())

with torch.inference_mode():
    y_pred_new = model_1(X_test)

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
