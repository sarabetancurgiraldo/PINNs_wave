#%% 
#%% Imports
import numpy as np
from fns import plot_initial_condition, animation_pde, plot_colormap, plot_error, NeuralNetwork, data_random_idx, save_model
import torch
from torchsummary import summary
from torch.optim import Adam
import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.nn.init as init


#%% Functions definition
def u_0(x):
    """
    dirac Initial condition. Height 1, width 0.1, centered at 0.5.
    """
    if x>=0 and x<0.45:
        return 0
    elif x>=0.45 and x<0.5:
        return 20*(x-0.45)
    elif x>=0.5 and x<0.55:
        return 20*(0.55-x)
    elif x>=0.55 and x<=1:
        return 0


def f_periodic_extension(x):
    if (x // 1) % 2 == 1:
        return -u_0(x % 1)
    else:
        return u_0(x % 1)


def u_dalembert(x, t):
    return (f_periodic_extension(x - t) + f_periodic_extension(x + t)) / 2


#%% Define parameters and grid

x_min, x_max = 0, 1 
t_0, n_t = 0, 1000 # time steps
c = 1
dx = 0.01
dt = 0.1 * dx / c
t_max = n_t * dt
x_points = np.linspace(x_min, x_max, int((x_max - x_min) / dx) + 1)
t_points = np.linspace(0, t_max, n_t + 1)


#%% Evaluate inital condition

u_0_values = np.array([u_0(x) for x in x_points])

# plot initial condition
plot_initial_condition(x_points, u_0_values)

#%%


#################################################################################
####### Task 1: Analytical solution using d'Alembert and odd extension ##########
#################################################################################


#%% Analytical solution

u_exact = np.zeros((len(x_points), len(t_points)))
for x_index, x_point in enumerate(x_points):
    for t_index, t_point in enumerate(t_points):
        u_exact[x_index, t_index] = u_dalembert(x_point, t_point)

# Plot solution
animation_pde(x_points, t_points, u_exact, x_min, x_max, n_t)
plot_colormap(u_exact, x_min, x_max, t_0, t_max, "Analytical Solution $u(x, t)$")

#%%


####################################################################################
####### Task 2: Numerical solution using finite differences: central diff ##########
####################################################################################


#%% Numerical solution 

u_fd = np.zeros((len(x_points), len(t_points)))
u_fd[:, 0] = u_exact[:, 0]
u_fd[:, 1] = u_exact[:, 1]  # option to solve the lack of data for the derivative at t=0
# u_fd[:, 1] = u_exact[:, 0]  # option to solve the lack of data for the derivative at t=0

for t_step in range(1, n_t):
    u_fd[1:-1, t_step + 1] = 2 * u_fd[1:-1, t_step] - u_fd[1:-1, t_step - 1] + (c ** 2) * (dt ** 2) * (u_fd[2:, t_step] - 2 * u_fd[1:-1, t_step] + u_fd[:-2, t_step]) / (dx ** 2)

# Plot numerical solution
animation_pde(x_points, t_points, u_fd, x_min, x_max, n_t)
plot_colormap(u_fd, x_min, x_max, t_0, t_max, "Numerical Solution $u(x, t)$")

#%% Plot error
plot_error(t_points, u_exact, u_fd, label='L2 error: Numerical vs Analytical solution')


#%%


###########################################################################
####### Task 3: PINN solution with data from analytical solution ##########
###########################################################################


from PINN_froward import train, plot_losses
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%% PINN solution


# # Choose data points
# x_train = []
# data_train = []
# idx_x = []
# for i in range(1, 34):
#     idx = i*3  # Every 3rd point
#     idx_x.append(idx)
#     x_train.append(x_points[idx])
#     data_train.append(u_exact[idx, 1:])
# t_train = t_points[1:]
'''
# mesh grid for training. choose points in space and time randomly (from the previous evaluated analytical solution)
Ns_x, Ns_t = 50, 200  # number of training points

# x_train, t_train, data_train, idx_x, idx_t = data_random_idx(x_points, Ns_x, u_exact, t_points[1:], Ns_t)
x_train, _, data_train, idx_x, _ = data_random_idx(x_points, Ns_x, u_exact)

idx_t = [i*4 for i in range(len(t_points[1:]) // 8)]  # every 4th point in time
t_train = t_points[idx_t]  # time points

# convert to torch tensors
x_train_t = torch.tensor(x_train, dtype=torch.float32)
t_train_t = torch.tensor(t_train, dtype=torch.float32)
data_train_t = torch.tensor(data_train, dtype=torch.float32)

# flatten the input for the model and make dimensions match 
# we don't need this now because we have the same number of points in x and t
x_train_tf, t_train_tf = torch.meshgrid(x_train_t.squeeze(), t_train_t.squeeze(), indexing='ij') 
x_train_tf = x_train_tf.reshape(-1, 1) # flattens the grid
t_train_tf = x_train_tf.reshape(-1, 1) # flattens the grid
data_train_tf = data_train_t.reshape(-1, 1)  # Reshape to match the input shape for the model

# Data for IC and BC
data_train_ic = u_exact[idx_x, 0]  # Initial condition data points
data_train_ic_dt = u_exact[idx_x, 1]  # Initial condition data points on "derivative"

data_train_bc_left = u_exact[0, :]  # Boundary condition data points at x=0
data_train_bc_right = u_exact[-1, :]  # Boundary condition data points at x=1
'''


# Number of points
N_f = 50000  # collocation (residual) points, not necessarily evenly spaced, and len(x_points) x len(t_points)
N_ic = 100  # initial condition points, not necessarily evenly spaced, and len(x_points) 
N_bc = 100  # boundary condition points, not necessarily evenly spaced, and len(t_points)


# Collocation points: interior of space-time domain
x_f = torch.rand((N_f, 1), dtype=torch.float32) * (x_max - x_min) + x_min
t_f = torch.rand((N_f, 1), dtype=torch.float32) * (t_max - t_0) + t_0

# IC: at t=0 we set u(x, 0) = u_0 and u_t(x, 0) = u_t_0
x_ic = torch.rand((N_ic, 1), dtype=torch.float32) * (x_max - x_min) + x_min
t_ic = torch.zeros_like(x_ic)
u_ic = np.interp(x_ic,x_points,u_exact[:, 0])  # Using the analytical solution for initial condition
u_t_ic = np.interp(x_ic,x_points,(u_exact[:, 1]-u_exact[:, 0])/dt)  # Using the analytical solution for initial condition
# otherwise, we could use u_fd[:, 1] - u_fd[:, 0] / dt
# or ignore u_t_ic and set it to zero
 
# BC: at x=0 and x=1 we set u(0, t) = 0 and u(1, t) = 0 (Dirichlet type)
t_bc = torch.rand((N_bc, 1), dtype=torch.float32) * (t_max - t_0) + t_0
x_bc_left  = torch.zeros_like(t_bc)
x_bc_right = torch.ones_like(t_bc)
u_bc_left  = np.interp(t_bc, t_points, u_exact[0, :])  # u(0, t)
u_bc_right = np.interp(t_bc, t_points, u_exact[-1, :])  # u(1, t)


# to torch tensors
data_train_ic_t = torch.tensor(u_ic, dtype=torch.float32)#.reshape(-1, 1)
data_train_ic_dt_t = torch.tensor(u_t_ic, dtype=torch.float32)#.reshape(-1, 1)
data_train_bc_left_t = torch.tensor(u_bc_left, dtype=torch.float32)#.reshape(-1, 1)
data_train_bc_right_t = torch.tensor(u_bc_right, dtype=torch.float32)#.reshape(-1, 1)

# move tensors to device
x_train_tf = x_f.to(device) 
t_train_tf = t_f.to(device)
x_ic = x_ic.to(device)
t_ic = t_ic.to(device)
t_bc = t_bc.to(device)
x_bc_left = x_bc_left.to(device)
x_bc_right = x_bc_right.to(device)

# data_train_tf = data_train_tf.to(device)
data_train_ic_t = data_train_ic_t.to(device)
data_train_ic_dt_t = data_train_ic_dt_t.to(device)
data_train_bc_left_t = data_train_bc_left_t.to(device)
data_train_bc_right_t = data_train_bc_right_t.to(device)


''''
# to torch tensors
data_train_ic_t = torch.tensor(data_train_ic, dtype=torch.float32)#.reshape(-1, 1)
data_train_ic_dt_t = torch.tensor(data_train_ic_dt, dtype=torch.float32)#.reshape(-1, 1)
data_train_bc_left_t = torch.tensor(data_train_bc_left, dtype=torch.float32)#.reshape(-1, 1)
data_train_bc_right_t = torch.tensor(data_train_bc_right, dtype=torch.float32)#.reshape(-1, 1) 

# move tensors to device
x_train_tf = x_train_tf.to(device) 
t_train_tf = t_train_tf.to(device)
data_train_tf = data_train_tf.to(device)
data_train_ic_t = data_train_ic_t.to(device)
data_train_ic_dt_t = data_train_ic_dt_t.to(device)
data_train_bc_left_t = data_train_bc_left_t.to(device)
data_train_bc_right_t = data_train_bc_right_t.to(device)
'''

#%% ######## Define the neural network model ########

# layers = [2, 64, 64, 1]  # Input layer (x, t), two hidden layers with 64 neurons each, output layer (u)
layers = [2, 64, 64, 1]  # Input layer (x, t), two hidden layers with 64 neurons each, output layer (u)
model_wave_fwd = NeuralNetwork(layers).to(device)

# don't really need this
# TODO: maybe remove 
# summary(model_wave_fwd, input_size=(2,)) # This will show the model summary for a single input tensor of shape (2,)

# Train the model
epochs=100000
l_pde, l_ic, l_ic_t, l_bc_l, l_bc_r = 1e-2, 1.0, 1e-2, 1e-2, 1e-2
lr = 1e-2
lr_lambdas = 1e-2
# Define optimizer
optimizer = Adam(model_wave_fwd.parameters(), lr=lr)

# history_fwd = train(model_wave_fwd, optimizer, epochs=epochs, print_every=500,
#                     x_f=x_train_tf, t_f=t_train_tf, u_ic=data_train_ic_t, u_t_ic=data_train_ic_dt_t,
#                     u_bc_left=data_train_bc_left_t, u_bc_right=data_train_bc_right_t,
#                     l_pde=l_pde, l_ic=l_ic, l_ic_t=l_ic_t, l_bc_l=l_bc_l, l_bc_r=l_bc_r)

# history_fwd = train(model_wave_fwd, optimizer, epochs=epochs, print_every=500,
#                     x_f=x_train_tf, t_f=t_train_tf, u_ic=data_train_ic_t, u_t_ic=data_train_ic_dt_t,
#                     u_bc_left=data_train_bc_left_t, u_bc_right=data_train_bc_right_t,
#                     flag_dual=True, l_r_lambdas=lr_lambdas, N_dual=100)


history_fwd = train(model_wave_fwd, optimizer, epochs=epochs, print_every=500,
                    x_f=x_train_tf, t_f=t_train_tf, x_ic=x_ic, t_ic=t_ic,
                    x_bc_left=x_bc_left, x_bc_right=x_bc_right, t_bc=t_bc,
                    u_ic=data_train_ic_t, u_t_ic=data_train_ic_dt_t,
                    u_bc_left=data_train_bc_left_t, u_bc_right=data_train_bc_right_t,
                    flag_dual=True, l_r_lambdas=lr_lambdas, N_dual=10)



# Save and plot
save_model(model_wave_fwd, f'models//PINN_physics//model_wave_fwd_epoch{epochs}_weights{l_pde}_{l_ic}_{l_ic_t}_{l_bc_l}_{l_bc_r}_lr_{lr}.pth')
plot_losses(history_fwd)

#why do we need this?
model_wave_fwd.load_state_dict(torch.load(f'models//PINN_physics//model_wave_fwd_epoch{epochs}_weights{l_pde}_{l_ic}_{l_ic_t}_{l_bc_l}_{l_bc_r}_lr_{lr}.pth'))

#%%
### Evaluate the model on all points in the grid
x_points_t = torch.tensor(x_points, dtype=torch.float32).to(device)
t_points_t = torch.tensor(t_points, dtype=torch.float32).to(device)
x_points_t_, t_points_t_ = torch.meshgrid(x_points_t.squeeze(), t_points_t.squeeze(), indexing='ij')
x_points_t_ = x_points_t_.reshape(-1, 1) # flattens the grid
t_points_t_ = t_points_t_.reshape(-1, 1) # flattens the grid

# Evaluate the model (inference)
model_wave_fwd.eval()
with torch.no_grad():
    # u_eval_fwd = model_wave_fwd(torch.cat((x_points_t, t_points_t), dim=1)).cpu().numpy().reshape(len(x_points_t), len(t_points_t))  # Reshape to (Nx, Nt) for plotting
    u_eval_fwd = model_wave_fwd(torch.cat((x_points_t_, t_points_t_), dim=1)).cpu().numpy().reshape(len(x_points_t), len(t_points_t))  # Reshape to (Nx, Nt) for plotting

# TODO: create function

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
#TODO: t goes forward (x-axis)
# PINN solution
im0 = axes[0].imshow(
    u_eval_fwd.T, 
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
plt.savefig(f'models//PINN_physics//model_wave_fwd_epoch{epochs}_weights{l_pde}_{l_ic}_{l_ic_t}_{l_bc_l}_{l_bc_r}_lr_{lr}.png')

# Plot initial condition
u_0_eval_PINN = u_eval_fwd[:, 0]  # Initial condition at t=0

plt.figure(figsize=(8, 4))
plt.plot(x_points, u_0_eval_PINN, label='PINN solution', color='yellow')
plt.plot(x_points, u_0_values, label='Analytical solution', color='blue')
plt.legend()
plt.title('Dirac Initial Condition $u_0(x)$')
plt.xlabel('x')
plt.ylabel('$u_0(x)$')
plt.savefig(f'models//PINN_physics//IC_fwd_epoch{epochs}_weights{l_pde}_{l_ic}_{l_ic_t}_{l_bc_l}_{l_bc_r}_lr_{lr}.png')

# N_ic = 100  # initial condition points, not necessarily evenly spaced, and len(x_points) 
# N_bc = 100  # boundary condition points, not necessarily evenly spaced, and len(t_points)



# %% 


################################################################################
####### Task 4: Domain and Data Sampling for WavePINN with 19 Sensors ##########
################################################################################


from PINN_data import train, plot_losses

#%%
# Domain and Data Sampling
# 19 sensors evenly spread (every 5*dx)

# Choose data points
x_grid = []
u_sensors = []
for i in range(1, 20):
    idx = i*5  # Every 5th point
    x_grid.append(x_points[idx])
    u_sensors.append(u_exact[idx, 1:])
t_grid = t_points[1:]

# Convert to tensors and move to device
x_grid = torch.tensor(x_grid, dtype=torch.float32)
t_grid = torch.tensor(t_grid, dtype=torch.float32)
u_sensors = torch.tensor(np.array(u_sensors), dtype=torch.float32)


x_grid_tf, t_grid_tf = torch.meshgrid(x_grid.squeeze(), t_grid.squeeze(), indexing='ij') 
x_grid_tf = x_grid_tf.reshape(-1, 1) # flattens the grid
t_grid_tf = t_grid_tf.reshape(-1, 1) # flattens the grid
u_sensors_tf = u_sensors.reshape(-1, 1)  # Reshape to match the input shape for the model

# move tensors to device
x_grid_tf = x_grid_tf.to(device)
t_grid_tf = t_grid_tf.to(device)
u_sensors_tf = u_sensors_tf.to(device)

# layers = [2, 64, 64, 1]  # Input layer (x, t), two hidden layers with 64 neurons each, output layer (u)
model_wave_sensors = NeuralNetwork(layers).to(device)

# Train the model
epochs=20000
l_pde, l_data = 1e-2, 1.0
lr = 1e-2
# Define optimizer
optimizer = Adam(model_wave_sensors.parameters(), lr=lr)

history_sensors = train(model_wave_sensors, optimizer, epochs=epochs, print_every=500,
                        x_f=x_train_tf, t_f=t_grid_tf, 
                        x_grid=x_grid_tf, t_grid=t_grid_tf, 
                        u_data=u_sensors_tf,
                        l_pde=l_pde, l_data=l_data)

# Save and plot
save_model(model_wave_sensors, f'models//PINN_sensors//model_epoch{epochs}_weights{l_pde}_{l_data}_lr_{lr}.pth')
plot_losses(history_sensors)

#why do we need this?
model_wave_sensors.load_state_dict(torch.load(f'models//PINN_sensors//model_epoch{epochs}_weights{l_pde}_{l_data}_lr_{lr}.pth'))

#%%
### Evaluate the model on all points in the grid
# x_points_t = torch.tensor(x_points, dtype=torch.float32)
# t_points_t = torch.tensor(t_points, dtype=torch.float32)
# x_points_t, t_points_t = torch.meshgrid(x_points_t.squeeze(), t_points_t.squeeze(), indexing='ij')
# x_points_t = x_points_t.reshape(-1, 1) # flattens the grid
# t_points_t = t_points_t.reshape(-1, 1) # flattens the grid

# Evaluate the model (inference)
model_wave_sensors.eval()
with torch.no_grad():
    # u_eval_sensors = model_wave_sensors(torch.cat((x_points_t, t_points_t), dim=1)).cpu().numpy().reshape(len(x_points_t), len(t_points_t))  # Reshape to (Nx, Nt) for plotting
    u_eval_sensors = model_wave_sensors(torch.cat((x_points_t_, t_points_t_), dim=1)).cpu().numpy().reshape(len(x_points_t), len(t_points_t))  # Reshape to (Nx, Nt) for plotting

# TODO: create function

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
#TODO: t goes forward (x-axis)
# PINN solution
im0 = axes[0].imshow(
    u_eval_sensors.T, 
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
plt.savefig(f'models//PINN_sensors//model_epoch{epochs}_weights{l_pde}_{l_data}_lr_{lr}.png')

# Plot initial condition
u_0_eval_PINN = u_eval_sensors[:, 0]  # Initial condition at t=0

plt.figure(figsize=(8, 4))
plt.plot(x_points, u_0_eval_PINN, label='Initial Condition $u_0(x)$', color='blue')
plt.title('Dirac Initial Condition')
plt.xlabel('x')
plt.ylabel('$u_0(x)$')
plt.savefig(f'models//PINN_sensors//model_epoch{epochs}_weights{l_pde}_{l_data}_lr_{lr}.png')


#%% 

#############################################################################################
####### Task 5: Domain and Data Sampling for WavePINN with randomly placed Sensors ##########
#############################################################################################


#%%

# Domain and Data Sampling
# mesh grid for training. choose points in space randomly (from the previous evaluated analytical solution)
# Ns sensors randomly spread 

Ns = 19  # Number of sensors

x_train_rand, _, data_train_rand, idx_x_rand, _ = data_random_idx(x_points, Ns, u_exact)

# convert to torch tensors
x_train_rand_t = torch.tensor(x_train_rand, dtype=torch.float32)
t_train_rand_t = torch.tensor(t_grid, dtype=torch.float32)

x_grid_rand_tf, t_grid_rand_tf = torch.meshgrid(x_train_rand_t.squeeze(), t_train_rand_t.squeeze(), indexing='ij') 
x_in_rand_tf = x_grid_rand_tf.reshape(-1, 1) # flattens the grid
t_in_rand_tf = t_grid_rand_tf.reshape(-1, 1) # flattens the grid
u_sensors_rand_tf = data_train_rand.reshape(-1, 1)  # Reshape to match the input shape for the model

# move tensors to device
x_in_rand_tf = x_in_rand_tf.to(device)
t_in_rand_tf = t_in_rand_tf.to(device)
u_sensors_tf = u_sensors_rand_tf.to(device)

model_wave_random = NeuralNetwork(layers).to(device)

# Train the model
epochs=20000
l_pde, l_data = 1e-2, 1.0
lr = 1e-2
# Define optimizer
optimizer = Adam(model_wave_random.parameters(), lr=lr)

history_random = train(model_wave_random, optimizer, epochs=epochs, print_every=500,
                        x_f=x_train_tf, t_f=t_grid_tf, 
                        # x_f=x_in_rand_tf, t_f=t_in_rand_tf, 
                        x_grid=x_in_rand_tf, t_grid=t_in_rand_tf, 
                        u_data=u_sensors_tf,
                        l_pde=l_pde, l_data=l_data)

# Save and plot
save_model(model_wave_random, f'models//PINN_random//model_epoch{epochs}_weights{l_pde}_{l_data}_lr_{lr}.pth')
plot_losses(history_random)

#why do we need this?
model_wave_random.load_state_dict(torch.load(f'models//PINN_random//model_epoch{epochs}_weights{l_pde}_{l_data}_lr_{lr}.pth'))

#%%
### Evaluate the model on all points in the grid
# x_points_t = torch.tensor(x_points, dtype=torch.float32)
# t_points_t = torch.tensor(t_points, dtype=torch.float32)
# x_points_t, t_points_t = torch.meshgrid(x_points_t.squeeze(), t_points_t.squeeze(), indexing='ij')
# x_points_t = x_points_t.reshape(-1, 1) # flattens the grid
# t_points_t = t_points_t.reshape(-1, 1) # flattens the grid

# Evaluate the model (inference)
model_wave_random.eval()
with torch.no_grad():
    # u_eval_random = model_wave_random(torch.cat((x_points_t, t_points_t), dim=1)).cpu().numpy().reshape(len(x_points_t), len(t_points_t))  # Reshape to (Nx, Nt) for plotting
    u_eval_random = model_wave_random(torch.cat((x_points_t_, t_points_t_), dim=1)).cpu().numpy().reshape(len(x_points_t), len(t_points_t))  # Reshape to (Nx, Nt) for plotting

# TODO: create function

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
#TODO: t goes forward (x-axis)
# PINN solution
im0 = axes[0].imshow(
    u_eval_random.T, 
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
plt.savefig(f'models//PINN_random//model_epoch{epochs}_weights{l_pde}_{l_data}_lr_{lr}.png')

# Plot initial condition
u_0_eval_PINN = u_eval_random[:, 0]  # Initial condition at t=0

plt.figure(figsize=(8, 4))
plt.plot(x_points, u_0_eval_PINN, label='Initial Condition $u_0(x)$', color='blue')
plt.title('Dirac Initial Condition')
plt.xlabel('x')
plt.ylabel('$u_0(x)$')
plt.savefig(f'models//PINN_random//model_epoch{epochs}_weights{l_pde}_{l_data}_lr_{lr}.png')










#%%