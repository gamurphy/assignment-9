import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
from functools import partial

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Generate data with a circular decision boundary
def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        
        # Initialize weights and biases
        self.weights1 = np.random.randn(input_dim, hidden_dim)
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, output_dim)
        self.bias2 = np.zeros((1, output_dim))

        # For visualization
        self.activations = []
        self.gradients = []

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'sigmoid':
            return self.activation(x) * (1 - self.activation(x))
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)

    def forward(self, X):
        # Forward pass
        self.z1 = X.dot(self.weights1) + self.bias1
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1.dot(self.weights2) + self.bias2
        self.a2 = 1 / (1 + np.exp(-self.z2))  # Sigmoid output for binary classification
        self.activations.append((self.a1, self.a2))
        return self.a2

    def backward(self, X, y):
        # Compute gradients
        m = X.shape[0]
        dz2 = self.a2 - y
        dw2 = self.a1.T.dot(dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = dz2.dot(self.weights2.T) * self.activation_derivative(self.z1)
        dw1 = X.T.dot(dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases using gradient descent
        self.weights1 -= self.lr * dw1
        self.bias1 -= self.lr * db1
        self.weights2 -= self.lr * dw2
        self.bias2 -= self.lr * db2

        # Store gradients for visualization
        self.gradients.append((dw1, dw2))

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden features visualization
    hidden_features = mlp.activations[-1][0]  # The hidden layer output from the latest step
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_xlabel('Neuron 1')
    ax_hidden.set_ylabel('Neuron 2')
    ax_hidden.set_zlabel('Neuron 3')
    
    # Optionally, plot a decision hyperplane in hidden space
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    plane_z = -(mlp.weights2[0, 0] * xx + mlp.weights2[1, 0] * yy + mlp.bias2[0, 0]) / (mlp.weights2[2, 0])
    ax_hidden.plot_surface(xx, yy, plane_z, alpha=0.3, color='brown')

    # Decision boundary in the input space
    ax_input.set_title(f"Input Space at Step {frame * 10}")
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')

    # Gradient visualization
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")
    dw1, dw2 = mlp.gradients[-1]
    
    # Visualize input to hidden connections
    input_nodes = [(-0.2, 0.8), (0.2, 0.8)]
    hidden_nodes = [(0.3, 0.5), (0.5, 0.5), (0.7, 0.5)]
    output_node = (0.5, 0.2)
    
    # Normalize gradient values for edge thickness
    max_grad_dw1 = np.max(np.abs(dw1))
    max_grad_dw2 = np.max(np.abs(dw2))

    for i, (x_in, y_in) in enumerate(input_nodes):
        for j, (x_hid, y_hid) in enumerate(hidden_nodes):
            weight = dw1[i, j]
            line_thickness = 1 + abs(weight) / max_grad_dw1 * 5  # Normalize and scale thickness
            ax_gradient.plot([x_in, x_hid], [y_in, y_hid], 'purple', linewidth=line_thickness)

    for j, (x_hid, y_hid) in enumerate(hidden_nodes):
        line_thickness = 1 + abs(dw2[j, 0]) / max_grad_dw2 * 5  # Normalize and scale thickness
        ax_gradient.plot([x_hid, output_node[0]], [y_hid, output_node[1]], 'purple', linewidth=line_thickness)

    # Draw the nodes
    for (x, y) in input_nodes + hidden_nodes + [output_node]:
        ax_gradient.scatter(x, y, s=300, color='blue')

    # Label nodes
    ax_gradient.text(input_nodes[0][0], input_nodes[0][1] + 0.05, 'x1', ha='center')
    ax_gradient.text(input_nodes[1][0], input_nodes[1][1] + 0.05, 'x2', ha='center')
    for idx, (x, y) in enumerate(hidden_nodes):
        ax_gradient.text(x, y + 0.05, f'h{idx + 1}', ha='center')
    ax_gradient.text(output_node[0], output_node[1] - 0.05, 'y', ha='center')

    # Add step text on the figure
    ax_input.text(-2.5, 2.5, f"Step: {frame * 10}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Visualization function
def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
