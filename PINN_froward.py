# %%
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam

# Define residual functions

# PDE residual function
def pde(model, x, t, c):
    x.requires_grad_(True) # Enable gradient tracking for x (True if gradients need to be computed for this tensor)
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

# IC_0 residual
def ic_0(model, x, t, u_ic):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(torch.cat((x, t), dim=1))

    residual = u - u_ic
    return residual

# IC_t_0 residual (time derivative at initial condition)
def ic_t_0(model, x, t, u_t_ic):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(torch.cat((x, t), dim=1))
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    residual = u_t - u_t_ic
    return residual

# Left BC residual
def bc_left(model, x, t, u_bc_left):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(torch.cat((x, t), dim=1))

    residual = u - u_bc_left
    return residual

# Right BC residual
def bc_right(model, x, t, u_bc_right):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(torch.cat((x, t), dim=1))

    residual = u - u_bc_right
    return residual

def train(model, optimizer, epochs, print_every,
          x_f, t_f, x_ic, t_ic, x_bc_left, x_bc_right, t_bc,
          u_ic, u_t_ic, u_bc_left, u_bc_right,
          l_pde=None, l_ic=None, l_ic_t=None, l_bc_l=None, l_bc_r=None,
          flag_dual=None, l_r_lambdas=None, N_dual=10):
        #   x_f, t_f, x_ic, t_ic, u_ic, u_t_ic,
        #   x_bc_left, x_bc_right, t_bc,
        #   u_bc_left, u_bc_right,
        #   l_pde, l_ic, l_ic_t, l_bc_l, l_bc_r):

    if flag_dual is not None:
        l_pde = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        l_ic = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        l_ic_t = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        l_bc_l = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        l_bc_r = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        dual_params = [l_pde, l_ic, l_ic_t, l_bc_l, l_bc_r]
        optimizer_dual = Adam(dual_params, lr=l_r_lambdas)

    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        if flag_dual is not None and epoch % N_dual == 0:
            optimizer_dual.zero_grad()
            loss, _ = compute_loss(model, x_f, t_f, x_ic, t_ic,
                                   x_bc_left, x_bc_right, t_bc, 
                                   u_ic, u_t_ic, u_bc_left, u_bc_right,
                                   l_pde, l_ic, l_ic_t, l_bc_l, l_bc_r)
            loss = - loss
            loss.backward()
            optimizer_dual.step()

        total_loss, loss = compute_loss(model, x_f, t_f, x_ic, t_ic,
                                         x_bc_left, x_bc_right, t_bc,
                                         u_ic, u_t_ic, u_bc_left, u_bc_right,
                                         l_pde, l_ic, l_ic_t, l_bc_l, l_bc_r)

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Store history
        history.append([loss[0], 
                        loss[1], 
                        loss[2],
                        loss[3],
                        loss[4],
                        loss[5]])

        if epoch % print_every == 0:
            # print(f"[{epoch}] Total: {loss.item():.4e} | PDE: {loss_pde.item():.4e} | IC_u: {loss_ic_u.item():.4e} | IC_ut: {loss_ic_ut.item():.4e} | BC_l: {loss_bc_left.item():.4e} | BC_r: {loss_bc_right.item():.4e}")
            print(f'''[{epoch}] Total: {history[epoch][0]:.4e} PDE: {history[epoch][1]:.4e} IC_u: {history[epoch][2]:.4e} IC_ut:{history[epoch][3]:.4e} BC_l: {history[epoch][4]:.4e} BC_r: {history[epoch][5]:.4e}''')

    return history

def plot_losses(history):
    history = torch.tensor(history)
    plt.figure(figsize=(10, 6))
    plt.plot(history[:, 0], label="Total Loss")
    plt.plot(history[:, 1], label="PDE Loss")
    plt.plot(history[:, 2], label="IC (u) Loss")
    plt.plot(history[:, 3], label="IC (ut) Loss")
    plt.plot(history[:, 4], label="BC Loss Left")
    plt.plot(history[:, 5], label="BC Loss Right")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_loss(model, x_f, t_f, x_ic, t_ic,
                 x_bc_left, x_bc_right, t_bc,
                 u_ic, u_t_ic, u_bc_left, u_bc_right, 
                 l_pde, l_ic, l_ic_t, l_bc_l, l_bc_r):
    ## LOSSES:
    # loss_pde = torch.mean((pde(model, x_f, t_f, c=1.0))**2)
    # loss_ic_u = torch.mean((ic_0(model, x_f, t_f, u_ic))**2)
    # loss_ic_ut = torch.mean((ic_t_0(model, x_f, t_f, u_t_ic))**2)
    # loss_bc_left = torch.mean((bc_left(model, x_f, t_f, u_bc_left))**2)
    # loss_bc_right = torch.mean((bc_right(model, x_f, t_f, u_bc_right))**2)
    loss_pde = torch.mean((pde(model, x_f, t_f, c=1.0))**2)
    loss_ic_u = torch.mean((ic_0(model, x_ic, t_ic, u_ic))**2)
    loss_ic_ut = torch.mean((ic_t_0(model, x_ic, t_ic, u_t_ic))**2)
    loss_bc_left = torch.mean((bc_left(model, x_bc_left, t_bc, u_bc_left))**2)
    loss_bc_right = torch.mean((bc_right(model, x_bc_right, t_bc, u_bc_right))**2)

    # === Total loss ===
    loss = (l_pde * loss_pde + 
            l_ic * loss_ic_u + 
            l_ic_t * loss_ic_ut + 
            l_bc_l * loss_bc_left + 
            l_bc_r * loss_bc_right)

    return loss, (loss.item(), loss_pde.item(), loss_ic_u.item(), loss_ic_ut.item(), loss_bc_left.item(), loss_bc_right.item())
