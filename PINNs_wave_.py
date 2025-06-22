#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

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

x_lower, x_upper = 0, 1
t_0, n_t = 0, 800
c = 1
dx = 0.01
dt = 0.1 * dx / c
x_points = np.linspace(x_lower, x_upper, int((x_upper - x_lower) / dx) + 1)
t_points = np.linspace(0, n_t * dt, n_t + 1)

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