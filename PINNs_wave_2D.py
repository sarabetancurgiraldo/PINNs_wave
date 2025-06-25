#%% Import libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Load Pytorch Modules and device definition
import torch
import torch.nn as nn
import torch.nn.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set random seed for reproducibility
torch.manual_seed(789)


#%% Define 2D domain, parameters and grid

x_min, x_max = 0, 1 # [m]
y_min, y_max = 0, 1 # [m]
t_0, t_max = 0, 1 # [s] original t_max = 4. # Reduced for faster computation
c = 1
dx, dy = 0.01, 0.01  # Spatial step sizes [m]
dt = 0.2 * min(dx,dy) / c  # CFL condition
n_t = int((t_max-t_0) / dt)

x_points = np.arange(x_min, x_max + dx, dx)
y_points = np.arange(y_min, y_max + dy, dy)
t_points = np.arange(t_0, t_max + dt, dt)

Nx = len(x_points)
Ny = len(y_points)
Nt = len(t_points)

print(f"Domain: x: [{x_min}, {x_max}], y: [{y_min}, {y_max}], t: [{t_0}, {t_max}]")
print(f"Grid: Nx = {Nx}, Ny = {Ny}, Nt = {Nt}")
print(f"dx = {dx}, dy = {dy}, dt = {dt}")

#%% Define Initial Condition — a 2D bump

def u0(x, y):
    r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    if r < 0.1:
        return np.exp(-400 * r**2)
    return 0.0
'''
    if 0.4 <= x <= 0.6 and 0.4 <= y <= 0.6:
        return np.exp(-100 * ((x - 0.5)**2 + (y - 0.5)**2))
    return 0.0
'''
u_fd = np.zeros((Nx, Ny, Nt))  # u(x, y, t)

# Set initial condition
for i, x in enumerate(x_points):
    for j, y in enumerate(y_points):
        u_fd[i, j, 0] = u0(x, y)
        u_fd[i, j, 1] = u_fd[i, j, 0]  # Assume zero initial velocity

#%% Plot initial condition
plt.figure(figsize=(8, 4))
plt.imshow(u_fd[:, :, 0], extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis')
plt.colorbar(label='$u_0(x, y)$')
plt.title('Initial Condition $u_0(x, y)$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()  


#%%

for n in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u_fd[i, j, n + 1] = (
                2 * u_fd[i, j, n] - u_fd[i, j, n - 1] +
                (c ** 2 * dt ** 2) * (
                    (u_fd[i + 1, j, n] - 2 * u_fd[i, j, n] + u_fd[i - 1, j, n]) / dx ** 2 +
                    (u_fd[i, j + 1, n] - 2 * u_fd[i, j, n] + u_fd[i, j - 1, n]) / dy ** 2
                )
            )

## 1:-1 internal points
## 2:,  are the points to the right
## 0:-2 are the points to the left

# %% plot the final time step
plt.figure(figsize=(8, 4))
plt.imshow(u_fd[:, :, -1], extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis')
plt.colorbar(label='$u(x, y, t_{max})$')
plt.title('Final Time Step $u(x, y, t_{max})$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



# %% animation of the FD solution
# Create a 2D animation of the FD solution  

fig, ax = plt.subplots()
frame = ax.imshow(u_fd[:, :, 0], extent=[x_min, x_max, y_min, y_max],
                  origin='lower', cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(frame, ax=ax)
ax.set_title("Time: 0.00 s")

def animate(n):
    frame.set_array(u_fd[:, :, n])
    ax.set_title(f"Time: {t_points[n]:.2f} s")
    return [frame]

anim = animation.FuncAnimation(
    fig, animate, frames=Nt, interval=50, blit=True
)


from IPython.display import HTML
HTML(anim.to_jshtml())


#%%
anim.save("fd_wave_2d.gif", writer='pillow', fps=20)

# %%
# Load Pytoch Modules and device definition
import torch
import torch.nn as nn
import torch.nn.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Define the neural network architecture
class WavePINN(nn.Module):
    def __init__(self, layers):
        super(WavePINN, self).__init__()
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
    '''
    def forward(self, x, t):
        inputs = torch.cat((x, t), dim=1)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    '''
    def forward(self, input):
        # Expect input to be a tensor of shape (batch_size, 2): [x, t]
        for layer in self.layers:
            input = layer(input)
        return input


# %%
# model summary 
from torchsummary import summary

# Define the model with the specified architecture
layers = [3, 64, 64, 1]  # Input layer (3 inputs: x, y and t, 2 hidden layers with 64 neurons each, and 1 output layer)
model = WavePINN(layers).to(device)

# summary(model, [(1, ), (1, ), (1, )])  # (x, y, t) shapes
summary(model, input_size=(3,)) # This will show the model summary for a single input tensor of shape (2,)

# %%
# Domain and Data Sampling
# (Domain is already defnied above)
# Define data and collocation points

from scipy.interpolate import RegularGridInterpolator

n_snapshots = 10                      # Number of time snapshots
points_per_snapshot = 1000            # Spatial points per snapshot
N_data = n_snapshots * points_per_snapshot

# Select fixed times (excluding t=0 and t_max if needed)
snapshot_times = np.linspace(t_0, t_max, n_snapshots + 2)[1:-1]  # exclude endpoints

# Preallocate lists
x_data_list, y_data_list, t_data_list = [], [], []

for t_snap in snapshot_times:
    x_rand = np.random.uniform(x_min, x_max, size=(points_per_snapshot,))
    y_rand = np.random.uniform(y_min, y_max, size=(points_per_snapshot,))
    t_fixed = np.full_like(x_rand, t_snap)

    x_data_list.append(x_rand)
    y_data_list.append(y_rand)
    t_data_list.append(t_fixed)

# Concatenate all (x, y, t) into arrays
x_data_np = np.concatenate(x_data_list)
y_data_np = np.concatenate(y_data_list)
t_data_np = np.concatenate(t_data_list)

# Interpolate u_fd at these (x, y, t)
interp_fd = RegularGridInterpolator((x_points, y_points, t_points), u_fd)
query_points = np.stack([x_data_np, y_data_np, t_data_np], axis=-1)
u_data_np = interp_fd(query_points)

# Convert to torch tensors
x_data = torch.tensor(x_data_np.reshape(-1, 1), dtype=torch.float32).to(device)
y_data = torch.tensor(y_data_np.reshape(-1, 1), dtype=torch.float32).to(device)
t_data = torch.tensor(t_data_np.reshape(-1, 1), dtype=torch.float32).to(device)
u_data = torch.tensor(u_data_np.reshape(-1, 1), dtype=torch.float32).to(device)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(x_data_np, y_data_np, t_data_np, c=u_data_np, cmap='viridis', s=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('t')
ax.set_title('Sparse Observations at Fixed Times from FD Solution')
# Rotate the view:
ax.view_init(elev=5, azim=135)  # change these values as you like
# Add colorbar
plt.colorbar(scatter, label='u(x, y, t)')
plt.show()



#%%
# === Collocation points for PDE residuals ===
# Generate collocation points in the domain
n_collocation = 5000  # Number of collocation points
x_f_np = np.random.uniform(x_min, x_max, size=(n_collocation,))
y_f_np = np.random.uniform(y_min, y_max, size=(n_collocation,))
t_f_np = np.random.uniform(t_0, t_max, size=(n_collocation,))   

# Fix shapes and convert to tensors
x_f = torch.tensor(x_f_np.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
y_f = torch.tensor(y_f_np.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
t_f = torch.tensor(t_f_np.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
# Convert numpy arrays to PyTorch tensors
# Note: requires_grad=True is set to allow gradients to be computed during training
x_f_np = torch.tensor(x_f_np.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
y_f_np = torch.tensor(y_f_np.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)      
t_f_np = torch.tensor(t_f_np.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)    


# %%
# Define residual functions

# PDE residuals 
def pde(model, x, y, t, c=1.0):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    u = model(torch.cat((x, y, t), dim=1))

    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]


    # Second derivatives
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    residual = u_tt - c**2 * (u_xx + u_yy)
    return residual

# data (snapshots) residuals
def data(model, x, y, t, obs_data):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    u = model(torch.cat((x, y, t), dim=1))

    residual = u - obs_data
    return residual


# %%
# Training Loop

from torch.optim import Adam

def train(model, optimizer, epochs, print_every,
          x_f, y_f, t_f, x_obs, y_obs, t_obs,
          l_pde, l_data, measurement):

    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        ## LOSSES:
        loss_pde = torch.mean((pde(model, x_f, y_f, t_f, c=1.0))**2)
        loss_data = torch.mean((data(model, x_obs, y_obs, t_obs, measurement))**2)  # if we want to use data loss 

        # === Total loss ===
        loss = (l_pde * loss_pde +
                l_data * loss_data)
        '''
        loss = (l_pde * loss_pde / loss_pde.detach() +
                l_ic * loss_ic_u / loss_ic_u.detach() +
                l_ic_t * loss_ic_ut / loss_ic_ut.detach() +
                l_bc_l * loss_bc_left / loss_bc_left.detach() +
                l_bc_r * loss_bc_right / loss_bc_right.detach()
                )'''

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Store history
        history.append([loss.item(), l_pde * loss_pde.item(), l_data * loss_data.item()])

        if epoch % print_every == 0:
            print(f"[{epoch}] Total: {loss.item():.4e} | PDE: {loss_pde.item():.4e} | Data: {loss_data.item():.4e}")

    return history


# %%
# Save Model and Plot Losses

import matplotlib.pyplot as plt
import os

saved_model_path = "wavepinn_model_2D.pth"

def save_model(model, path=saved_model_path):
    torch.save(model.state_dict(), path)
    print(f"[💾] Model saved to: {os.path.abspath(path)}")

def plot_losses(history):
    history = torch.tensor(history)
    plt.figure(figsize=(10, 6))
    plt.plot(history[:, 0], label="Total Loss")
    plt.plot(history[:, 1], label="PDE Loss")
    plt.plot(history[:, 2], label="Data Loss")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%
# Utility function to check gradient path

def check_gradients(u, inputs, names):
    """
    For each input tensor, checks whether the gradient of `u` with respect to that input is non-None.
    Prints diagnostic messages for debugging.

    Parameters:
    - u: scalar output from the model (or any function of inputs)
    - inputs: list of tensors to check (must have requires_grad=True)
    - names: list of strings (same length) naming each input
    """
    for input_tensor, name in zip(inputs, names):
        grad = torch.autograd.grad(
            u, input_tensor, 
            grad_outputs=torch.ones_like(u), 
            retain_graph=True, create_graph=True, 
            allow_unused=True
        )[0]
        if grad is None:
            print(f"[⚠️] Gradient w.r.t. '{name}' is None — check requires_grad!")
        else:
            print(f"[✅] Gradient w.r.t. '{name}' is OK: shape {grad.shape}")


# %%
# Small test inputs (on correct device)
x_test = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True).to(device)
y_test = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True).to(device)
t_test = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True).to(device)

# Forward pass
u_test = model(torch.cat((x_test, y_test, t_test), dim=1))

# Check gradient paths
check_gradients(u_test, [x_test, y_test, t_test], ["x", "y", "t"])


# %%
# Initialize model, optimizer, and training parameters

# Define model
# model = WavePINN(layers).to(device)
optimizer = Adam(model.parameters(), lr=1e-2)


# Now pass these to the train function
history = train(
    model, optimizer, epochs=100000, print_every=500,
    x_f=x_f_np, y_f=y_f_np, t_f=t_f_np,
    x_obs=x_data, y_obs=y_data, t_obs=t_data,
    l_pde=1e-1, l_data=1.0, measurement=u_data
)

# Save and plot
save_model(model)
plot_losses(history)


# %%
# model reloading (to avoid retraining, if needed)
# model = WavePINN(layers).to(device)
model.load_state_dict(torch.load(saved_model_path))
# model.eval()  # Set model to evaluation mode


# %%
# Final evaluation and visualization
# Generate evaluation points
Nx_eval, Ny_eval, Nt_eval = 101, 101, 101

x_eval = torch.linspace(x_min, x_max, Nx_eval).to(device)
y_eval = torch.linspace(y_min, y_max, Ny_eval).to(device)
t_eval = torch.linspace(t_0, t_max, Nt_eval).to(device)

# 3D meshgrid: shape (Nx, Ny, Nt)
X, Y, T = torch.meshgrid(x_eval, y_eval, t_eval, indexing='ij')

# Flatten into (Nx * Ny * Nt, 1) for input to model
x_input = X.reshape(-1, 1)
y_input = Y.reshape(-1, 1)
t_input = T.reshape(-1, 1)

# Concatenate for input to PINN model: (Nx * Ny * Nt, 3)
input_tensor = torch.cat((x_input, y_input, t_input), dim=1)


# Evaluate the model (inference)
model.eval()
with torch.no_grad():
    u_eval = model(input_tensor).cpu().numpy().reshape(Nx_eval, Ny_eval, Nt_eval)  # Reshape to (Nx, Ny, Nt) for plotting

#%%
# Plot the results at a specific time slice
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Choose middle time slice
k = Nt_eval // 2
t_mid = t_eval[k].item()
k_fd = np.argmin(np.abs(t_points - t_mid))

# PINN solution
im0 = axes[0].imshow(
    u_eval[:, :, k], 
    extent=(x_min, x_max, y_min, y_max),
    aspect='auto', 
    origin='lower',
    cmap='viridis', 
    interpolation='nearest',
    vmin=-0.2, vmax=0.2)  # u_eval.T: transpose to get x horizontal, t vertical
axes[0].set_title('PINN Solution')
axes[0].set_xlabel('x (space)')
axes[0].set_ylabel('y (space)')
fig.colorbar(im0, ax=axes[0], orientation='vertical', label='u(x, y)')

# FD Solution
im1 = axes[1].imshow(
    u_fd[:, :, k_fd],
    extent=(x_min, x_max, y_min, y_max),
    aspect='auto',
    origin='lower',
    cmap='viridis',
    interpolation='nearest',
    vmin=-0.2, vmax=0.2)  # u_fd.T: transpose to get x horizontal, t vertical
axes[1].set_title('FD Solution')
axes[1].set_xlabel('x (space)')
axes[1].set_ylabel('y (space)')
fig.colorbar(im1, ax=axes[1], orientation='vertical', label='u(x, y)')

plt.tight_layout()
plt.show()

#%% Plot the results at on a (x,t) slice, y=0.5
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Choose middle time slice
k = Ny_eval // 2
y_mid = y_eval[k].item()
k_fd = np.argmin(np.abs(y_points - y_mid))

# PINN solution
u_plot = u_eval[:, k, :]  # Extract the slice at y_mid
im0 = axes[0].imshow(
    u_plot.T, 
    extent=(x_min, x_max, t_0, t_max),
    aspect='auto', 
    origin='lower',
    cmap='viridis', 
    interpolation='nearest',
    vmin=-0.2, vmax=0.2)  # u_eval.T: transpose to get x horizontal, t vertical
axes[0].set_title('PINN Solution')
axes[0].set_xlabel('x (space)')
axes[0].set_ylabel('t (time)')
fig.colorbar(im0, ax=axes[0], orientation='vertical', label='u(x, t)')

# FD Solution
im1 = axes[1].imshow(
    u_fd[:, k_fd, :],
    extent=(x_min, x_max, t_0, t_max),
    aspect='auto',
    origin='lower',
    cmap='viridis',
    interpolation='nearest',
    vmin=-0.2, vmax=0.2)  # u_fd.T: transpose to get x horizontal, t vertical
axes[1].set_title('FD Solution')
axes[1].set_xlabel('x (space)')
axes[1].set_ylabel('t (time)')
fig.colorbar(im1, ax=axes[1], orientation='vertical', label='u(x, t)')

plt.tight_layout()
plt.show()


# %%
