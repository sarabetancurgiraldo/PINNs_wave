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


#%% Function definition
def u_0(x):
    """
    "Dirac" Initial condition. Height 1, width 0.1, centered at 0.5.
    """
    if x>=0 and x<0.45:
        return 0
    elif x>=0.45 and x<0.5:
        return 20*(x-0.45)
    elif x>=0.5 and x<0.55:
        return 20*(0.55-x)
    elif x>=0.55 and x<=1:
        return 0


#%% Source time function (Gaussian)
# def u_0(x):
#     # Parameters for the Gaussian
#     a = 0.05  # width parameter (standard deviation)
#     x0 = 0.5    # center of the Gaussian
#     scale = 10
# 
#     # First derivative of a Gaussian
#     s = (x - x0)
#     return 1/scale * -s / (a ** 2) * math.exp(-s**2 / (2 * a**2))


#%% Define parameters and grid

x_min, x_max = 0, 1 # [m]
t_0, t_max = 0, 1 # [s] original t_max = 4. # Reduced for faster computation
c = 1
dx = 0.01
dt = 0.2 * dx / c
n_t = int((t_max-t_0) / dt)
x_points = np.linspace(x_min, x_max, int((x_max - x_min) / dx) + 1)
t_points = np.linspace(t_0, t_max, n_t + 1)

# Evaluate initial condition at x points
u_0_values = np.array([u_0(x) for x in x_points])

#%% Plot initial condition
plt.figure(figsize=(8, 4))
plt.plot(x_points, u_0_values, label='Initial Condition $u_0(x)$', color='blue')
plt.title('Dirac Initial Condition')
plt.xlabel('x')
plt.ylabel('$u_0(x)$')

#%% 

def f_periodic_extension(x):
    if (x // 1) % 2 == 1:
        return -u_0(x % 1)
    else:
        return u_0(x % 1)


def u_dalembert(x, t):
    return (f_periodic_extension(x - t) + f_periodic_extension(x + t)) / 2


u_exact = np.zeros((len(x_points), len(t_points)))
for x_index, x_point in enumerate(x_points):
    for t_index, t_point in enumerate(t_points):
        u_exact[x_index, t_index] = u_dalembert(x_point, t_point)

#%%

fig, ax = plt.subplots()
line, = ax.plot(x_points, u_exact[:, 0])
ax.set_xlim(x_min, x_max)  
ax.set_ylim(-1, 1)
def animate_exact(i):
    line.set_data((x_points, u_exact[:, i]))
    time_step = np.round(t_points[i], 1)
    plt.title('Time: ' + str(time_step))
    plt.xlabel('$x$')
    plt.ylabel('$u(x, ' + str(time_step) + ')$')
    return (line,)

anim = animation.FuncAnimation(fig, animate_exact, frames=np.arange(0, n_t, 40), interval=100, blit=True)
HTML(anim.to_jshtml())


#%%

u_fd = np.zeros((len(x_points), len(t_points)))
u_fd[:, 0] = u_exact[:, 0]
# u_fd[:, 1] = u_exact[:, 1]
u_fd[:, 1] = u_exact[:, 0]  # my numerical way of doing it, if I don't have analytical solution for t=1


for t_step in range(1, n_t):
    u_fd[1:-1, t_step + 1] = 2 * u_fd[1:-1, t_step] - u_fd[1:-1, t_step - 1] + (c ** 2) * (dt ** 2) * (u_fd[2:, t_step] - 2 * u_fd[1:-1, t_step] + u_fd[:-2, t_step]) / (dx ** 2)
## 1:-1 internal points
## 2:,  are the points to the right
## 0:-2 are the points to the left

# %%
fig, ax = plt.subplots()
line, = ax.plot(x_points, u_fd[:, 0])
ax.set_xlim(x_min, x_max)  
ax.set_ylim(-1, 1)
def animate_fd(i):
    line.set_data((x_points, u_fd[:, i]))
    time_step = np.round(t_points[i], 1)
    plt.title('Time: ' + str(time_step))
    plt.xlabel('$x$')
    plt.ylabel('$u(x, ' + str(time_step) + ')$')
    return (line,)

anim = animation.FuncAnimation(fig, animate_fd, frames=np.arange(0, n_t, 40), interval=100, blit=True)
HTML(anim.to_jshtml())

#  %% Plot error
error = np.zeros((len(t_points), 2))
for t_index in range(len(t_points)):
    error[t_index, 0] = np.linalg.norm(u_fd[:, t_index] - u_exact[:, t_index])

plt.plot(error[:, 0], label='Error: FD wrt. analytical', color='red')
# Plot styling
plt.xlabel('time steps')
plt.ylabel('L2 error')
plt.legend()

# %%
# Load Pytoch Modules and device definition
import torch
import torch.nn as nn
import torch.nn.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
torch.manual_seed(789)

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
layers = [2, 64, 64, 1]  # Input layer (2 inputs: x and t, 2 hidden layers with 64 neurons each, and 1 output layer)
model = WavePINN(layers).to(device)

# summary(model, [(1, ), (1, )])  # (x, t) shapes
summary(model, input_size=(2,)) # This will show the model summary for a single input tensor of shape (2,)

# %%
# Domain and Data Sampling
# (Domain is already defnied above)
# Define BC, IC, and collocation points

# Number of points
N_f = 5000  # collocation (residual) points, not necessarily evenly spaced, and len(x_points) x len(t_points)
# N_s = len(t_points)  # sensor points (inherited from the anaytical solution)


# Collocation points: interior of space-time domain
x_f = torch.rand((N_f, 1), dtype=torch.float32) * (x_max - x_min) + x_min
t_f = torch.rand((N_f, 1), dtype=torch.float32) * (t_max - t_0) + t_0


# Sensors data (Dirichlet type data)
n_rec = 5

# uniformly spaced receivers
# receivers = np.linspace(np.min(x_points), np.max(x_points), n_rec+2) # Sensors data (Dirichlet type data)
# receivers = receivers[1:-1] # remove first and last points to avoid including boundaries

# Randomly sampled receivers
receivers = np.random.uniform(x_min, x_max, size=(n_rec,))

observations = np.zeros((len(t_points), n_rec))

for j, t in enumerate(t_points):
    for i, x in enumerate(receivers):
        observations[j, i] = u_dalembert(x, t)

# alternative way to get observations
# for j in range(len(t_points)):
#    # Interpolate u_exact at receiver locations for this time step
#    observations[j, :] = np.interp(receivers, x_points, u_exact[:, j])

# Convert to torch tensors and move to device
x_f = x_f.to(device)
t_f = t_f.to(device)
receivers_torch = torch.tensor(receivers, dtype=torch.float32).to(device)  # shape: (n_rec,)
observations_torch = torch.tensor(observations, dtype=torch.float32).to(device)  # shape: (len(t_points), n_rec)

sorted_indices = np.argsort(receivers)
receivers = receivers[sorted_indices]
observations = observations[:, sorted_indices]  # reorder columns

X_obs, T_obs = np.meshgrid(receivers, t_points, indexing='ij')  # shape: (n_rec, n_t)


plt.figure(figsize=(8, 4))
plt.pcolormesh(T_obs, X_obs, observations.T, shading='auto', cmap='viridis')
plt.colorbar(label='u(x, t)')
plt.xlabel('Time t')
plt.ylabel('Receiver position x')
plt.title('Sparse Observations (pcolormesh)')
plt.show()

'''
plt.figure(figsize=(8,4))
plt.imshow(np.flipud(observations.T), aspect='auto')
plt.title('Map based on Sensor Observations (u(x, t))')
plt.xlabel('Time step #')
plt.ylabel('Receiver index #')
plt.colorbar()
plt.show()
'''

# Print shapes for verification
print(f"x_f shape: {x_f.shape}, t_f shape: {t_f.shape}")
print(f"observation shape: {observations_torch.shape}, receivers shape: {receivers_torch.shape}")
# Check if all tensors are on the correct device
print(f"Device: {x_f.device}, t_f device: {t_f.device}")
print(f"observation device: {observations_torch.device}, receivers device: {receivers_torch.device}")



# %%
# Define residual functions

# PDE residual function
def pde(model, x, t, c=1.0):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(torch.cat((x, t), dim=1))

    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    # Second derivatives
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    residual = u_tt - c**2 * u_xx
    return residual

# Sensors residual
def data(model, x, t, obs_data):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(torch.cat((x, t), dim=1))

    residual = u - obs_data
    return residual


# %%
# Training Loop

from torch.optim import Adam

def train(model, optimizer, epochs, print_every,
          x_f, t_f, x_obs, t_obs,
          l_pde, l_data, measurement):

    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        ## LOSSES:
        loss_pde = torch.mean((pde(model, x_f, t_f, c=1.0))**2)
        loss_data = torch.mean((data(model, x_obs, t_obs, measurement))**2)  # if we want to use data loss 

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

saved_model_path = "wavepinn_model_sensors_task_5_5sensors.pth"

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
t_test = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True).to(device)

# Forward pass
u_test = model(torch.cat((x_test, t_test), dim=1))

# Check gradient paths
check_gradients(u_test, [x_test, t_test], ["x", "t"])


# %%
# Initialize model, optimizer, and training parameters

# Define model
# model = WavePINN(layers).to(device)
optimizer = Adam(model.parameters(), lr=1e-2)

# Prepare observation input pairs and targets for data loss
# x_obs: (n_rec,), t_obs: (n_time_steps,)
x_obs_grid, t_obs_grid = torch.meshgrid(
    receivers_torch, torch.tensor(t_points, dtype=torch.float32).to(device), indexing='ij'
)  # both shape: (n_rec, n_time_steps)

# Flatten to (n_rec * n_time_steps, 1)
x_obs_flat = x_obs_grid.reshape(-1, 1)
t_obs_flat = t_obs_grid.reshape(-1, 1)
measurement_flat = observations_torch.T.reshape(-1, 1)  # observations_torch: (n_time_steps, n_rec) → (n_rec, n_time_steps) → flatten

# Now pass these to the train function
history = train(
    model, optimizer, epochs=100000, print_every=500,
    x_f=x_f, t_f=t_f,
    x_obs=x_obs_flat, t_obs=t_obs_flat,
    l_pde=1e-2, l_data=1.0, measurement=measurement_flat
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
Nx= 101  # Number of points in x
Nt = 1001  # Number of points in t
x_eval = torch.linspace(x_min, x_max, Nx).view(-1, 1).to(device)
t_eval = torch.linspace(t_0, t_max, Nt).view(-1, 1).to(device)
x_grid, t_grid = torch.meshgrid(x_eval.squeeze(), t_eval.squeeze(), indexing='ij') # shape (Nx, Nt)
x_input = x_grid.reshape(-1, 1) # flattens the grid
t_input = t_grid.reshape(-1, 1) # flattens the grid

# Evaluate the model (inference)
model.eval()
with torch.no_grad():
    u_eval = model(torch.cat((x_input, t_input), dim=1)).cpu().numpy().reshape(Nx, Nt)  # Reshape to (Nx, Nt) for plotting

# Reconstruct meshgrid for contours from evaluation range
x_vals = x_eval[:, 0].cpu().numpy()
t_vals = t_eval[:, 0].cpu().numpy()
X, T = np.meshgrid(x_vals, t_vals, indexing='ij')  # X: (Nx, Nt), T: (Nx, Nt)

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# PINN solution
im0 = axes[0].imshow(
    u_eval.T, 
    extent=(x_min, x_max, t_0, t_max), 
    aspect='auto', 
    origin='lower',
    cmap='viridis', 
    interpolation='nearest',
    vmin=-1.0, vmax=1.0)  # u_eval.T: transpose to get x horizontal, t vertical
axes[0].set_title('PINN Solution')
axes[0].set_xlabel('x (space)')
axes[0].set_ylabel('t (time)')
fig.colorbar(im0, ax=axes[0], orientation='vertical', label='u(x, t)')

# Analytical Solution
im1 = axes[1].imshow(
    u_exact.T,
    extent=(x_min, x_max, t_0, t_max),
    aspect='auto',
    origin='lower',
    cmap='viridis',
    interpolation='nearest',
    vmin=-1.0, vmax=1.0
)
axes[1].set_title('Analytical Solution')
axes[1].set_xlabel('x (space)')
fig.colorbar(im1, ax=axes[1], orientation='vertical', label='u(x, t)')

plt.tight_layout()
plt.show()


# %%
# some checks

print("u_exact.shape =", u_exact.shape)
print("X.shape =", X.shape)
print("T.shape =", T.shape)
# 
print("u_exact.shape =", u_exact.shape)
print("NaNs in u_exact:", np.isnan(u_exact).sum())
print("Infs in u_exact:", np.isinf(u_exact).sum())
print("min/max of u_exact:", np.nanmin(u_exact), np.nanmax(u_exact))
nan_indices = np.argwhere(np.isnan(u_exact))
print("NaN found at indices:", nan_indices)
# 
plt.imshow(u_exact.T, extent=(x_min, x_max, t_0, t_max), origin='lower', cmap='plasma')
plt.title("Analytical Solution (u_exact.T)")
plt.colorbar()
plt.show()

# %%

# %%
#Animation of the PINN solution

# 1. Create meshgrid
X, T = np.meshgrid(x_points, t_points, indexing='ij')  # Shape: (Nx, Nt)
X_flat = X.flatten()
T_flat = T.flatten()

# 2. Build input tensor for PINN
input_tensor = torch.tensor(np.stack([X_flat, T_flat], axis=1), dtype=torch.float32).to(device)

# 3. Predict with model
with torch.no_grad():
    u_pred_flat = model(input_tensor).cpu().numpy().flatten()

# 4. Reshape prediction
u_pinn = u_pred_flat.reshape(len(x_points), len(t_points))  # Shape: (Nx, Nt)

# 5. Set up animation
fig, ax = plt.subplots()
line, = ax.plot(x_points, u_pinn[:, 0])
# ax.set_xlim(x_points.min(), x_points.max())
# ax.set_ylim(u_pinn.min(), u_pinn.max())
ax.set_xlim(x_min, x_max)  
ax.set_ylim(-1, 1)

def animate_pinn(i):
    line.set_ydata(u_pinn[:, i])
    ax.set_title(f'Time: {t_points[i]:.2f}')
    return (line,)

anim = animation.FuncAnimation(
    fig, animate_pinn, frames=np.arange(0, len(t_points), 4), interval=100, blit=True
)

HTML(anim.to_jshtml())

# %%
