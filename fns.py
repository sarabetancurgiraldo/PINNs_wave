#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch.nn as nn
import torch.nn.init as init
import torch
import os


#%% Plot initial condition
def plot_initial_condition(x_grid, u_0):
    """
    Plots the initial condition u_0(x) on the given x_grid.

    Parameters:
    x_grid (numpy.ndarray): The grid of x values.
    u_0 (numpy.ndarray): The initial condition values corresponding to x_grid.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(x_grid, u_0, label='Initial Condition $u_0(x)$', color='blue')
    plt.title('Dirac Initial Condition')
    plt.xlabel('x')
    plt.ylabel('$u_0(x)$')
    plt.legend()
    plt.grid()
    plt.show()


def animation_pde(x, t, u, x_min, x_max, n_t):
    
    fig, ax = plt.subplots()
    line, = ax.plot(x, u[:, 0])
    ax.set_xlim(x_min, x_max)  
    ax.set_ylim(-1, 1)
    def animate_exact(i):
        line.set_data((x, u[:, i]))
        time_step = np.round(t[i], 1)
        plt.title('Time: ' + str(time_step))
        plt.xlabel('$x$')
        plt.ylabel('$u(x, ' + str(time_step) + ')$')
        return (line,)

    anim = animation.FuncAnimation(fig, animate_exact, frames=np.arange(0, n_t, 40), interval=100, blit=True)
    HTML(anim.to_jshtml())


def plot_colormap(u, x_min, x_max, t_0, t_max, title):
    plt.figure(figsize=(10, 6))
    # plt.imshow(u, extent=(t_0, t_max, x_min, x_max), origin='lower', cmap='plasma')
    plt.imshow(u.T, extent=(x_min, x_max, t_0, t_max), origin='lower', cmap='plasma')
    plt.title(title)
    # plt.xlabel('$t$')
    # plt.ylabel('$x$')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.colorbar()
    plt.show()


def plot_error(t, u1, u2, label):
    error = np.zeros((len(t), 2))
    for t_index in range(len(t)):
        error[t_index, 0] = np.linalg.norm(u1[:, t_index] - u2[:, t_index])

    plt.plot(error[:, 0], label=label, color='red')
    # Plot styling
    plt.xlabel('time steps')
    plt.ylabel('L2 error')
    plt.legend()


# Define neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.Tanh())
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.xavier_uniform_(m.weight)  # or xavier_normal_
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias) 

    def forward(self, input):
        # Expect input to be a tensor of shape (batch_size, 2): [x, t]
        for layer in self.layers:
            input = layer(input)
        return input

def data_random_idx(x, Ns_x, data, t=None, Ns_t=None):
    idx_x = np.random.choice(len(x), Ns_x, replace=False)
    if t is not None:
        idx_t = np.random.choice(len(t), Ns_t, replace=False)
    # Choose data points
    t_grid = []
    x_grid = []
    data_grid = []
    for i in idx_x:
        x_grid.append(x[i])
        if t is not None:
            for j in idx_t:
                t_grid.append(t[j])
                data_grid.append(data[i, j])
        else:
            # All data points for the given x and all t but IC
            data_grid.append(data[i, 1:])
            idx_t = 0
    return x_grid, t_grid, np.array(data_grid), idx_x, idx_t


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {os.path.abspath(path)}")









#%%











