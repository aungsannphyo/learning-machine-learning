from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, torch
from helper_functions import plot_decision_boundary

# make 1000 samples
n_samples = 1000
X, y = make_circles(n_samples,
                    noise=0.03, random_state=42)

# convert to pandas
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})

# visualize
circles.plot(kind="scatter", x="X1", y="X2", c="label", cmap=plt.cm.RdYlBu)
# plt.show()

# check input and output shape
print(X.shape, y.shape)

# device agnostic code
device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 20% of data for test 80% of data for train
    random_state=42,
    stratify=y  # keep the same ratio of labels in train and test
)


# building model
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # Create nn Linear Layers capable of handling the data
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=10, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1, bias=True),
        )

    # define forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


# create instance of model and send it to device
model_0 = CircleModelV0().to(device)
print(model_0.state_dict())

# make predictions
with torch.inference_mode():
    untrained_pred = model_0(X_test.to(device))
print(
    f"Length of predictions: {len(untrained_pred)} , Shape of predictions: {untrained_pred.shape}")
print(
    f"Length of test data: {len(X_test)} , Shape of test data: {X_test.shape}")
print(f"\nFirst 10 predictions:\n {untrained_pred[:10]}")
print(f"\nFirst 10 test labels:\n {y_test[:10]}")

# Setup loss function and optimizer
# sigmoid activation function built into the loss function
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# Calculate accuracy - out of 100 examples, how many the model gets right


def accuracy_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_pred) * 100


torch.manual_seed(42)
torch.mps.manual_seed(42)

# Train model
epochs = 1000
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    # Set model to training mode
    model_0.train()

    # Forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Calculate loss and accuracy
    # nn.BCEWithLogitsLoss() takes raw logits as input
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_pred, y_train)

    # Optimizer zero grad
    optimizer.zero_grad()

    # loss.backward() - perform backpropagation
    loss.backward()

    # optimizer.step() - perform gradient descent
    optimizer.step()

    # Set model to evaluation mode
    model_0.eval()

    # Forward pass
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(test_pred, y_test)

    # Print results
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

# plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train Data")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plot_decision_boundary(model_0, X_test, y_test)
plt.title("Test Data")
plt.show()
