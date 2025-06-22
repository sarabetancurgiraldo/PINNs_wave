#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt

#%% Function definition
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

#%% Define parameters and grid
dx = 1e-2
grid_x = np.linspace(0, 1, int(1/dx) + 1)
u_0_values = np.array([u_0(x) for x in grid_x])

#%% Plot initial condition
plt.figure(figsize=(8, 4))
plt.plot(grid_x, u_0_values, label='Initial Condition $u_0(x)$', color='blue')
plt.title('Dirac Initial Condition')
plt.xlabel('x')
plt.ylabel('$u_0(x)$')



#%%