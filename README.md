# PyTorch Neural Network Classification

This project demonstrates a neural network built with PyTorch for binary classification, using two different models with varying architectures and accuracies.

## Project Overview
- **Dataset**: Synthetic dataset with two input features.
- **Model Architectures**:
  - **Model_0**: Simple architecture with one hidden layer, achieving ~50% accuracy.
  - **Model**: Improved architecture with two hidden layers, achieving ~99% accuracy.
- **Training Process**: Stochastic Gradient Descent (SGD) optimizer and Binary Cross Entropy with Logits Loss.
- **Visualization**: Data distribution and decision boundary of both models.

## Installation
Ensure you have Python and PyTorch installed:
```bash
pip install torch matplotlib
```

## Data Visualization
We visualize the dataset before training, using Matplotlib to understand the decision boundary.
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Dataset Distribution")
    plt.show()

plot_data(x_train.cpu().numpy(), y_train.cpu().numpy())
```

## Model Architectures
### Model_0 (Baseline: 50% Accuracy)
```python
class Model_0(nn.Module):
    def __init__(self):
        super(Model_0, self).__init__()
        self.layer1 = nn.Linear(2, 5)
        self.output = nn.Linear(5, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.output(x)
        return x
```

### Model (Improved: 99% Accuracy)
```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden1 = nn.Linear(2, 4)
        self.hidden2 = nn.Linear(4, 2)
        self.output = nn.Linear(2, 1)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.output(x)
        return x
```

## Training Process
```python
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    y_logits = model(x_train).squeeze()
    loss = loss_fn(y_logits, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

## Results
| Model   | Test Accuracy |
|---------|--------------|
| Model_0 | 50%         |
| Model   | 99%         |

## Visualization of Decision Boundaries
We visualize the decision boundaries of both models after training to compare their effectiveness in classification.

```python
import numpy as np

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(grid).reshape(xx.shape)
    plt.contourf(xx, yy, torch.sigmoid(preds).cpu().numpy(), cmap='coolwarm', alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.show()

plot_decision_boundary(model, x_train.cpu().numpy(), y_train.cpu().numpy())
```

## Conclusion
- Model_0 struggles to classify data correctly (~50% accuracy).
- The improved Model, with additional hidden layers and activation functions, achieves ~99% accuracy.
- Visualizations confirm better decision boundaries in the improved model.

## Future Improvements
- Experiment with different activation functions.
- Tune hyperparameters (e.g., learning rate, batch size).
- Explore different optimizers (Adam, RMSprop).

## Author
**Sanchit**

