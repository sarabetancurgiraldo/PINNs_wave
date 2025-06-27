# %%
import torch
import matplotlib.pyplot as plt


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

# data residual
def data(model, x, t, u_data):
    u = model(torch.cat((x, t), dim=1))

    residual = u - u_data
    return residual



def train(model, optimizer, epochs, print_every,
          x_f, t_f, x_grid, t_grid, u_data,
          l_pde, l_data):

    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        ## LOSSES:
        loss_pde = torch.mean((pde(model, x_f, t_f, c=1.0))**2)
        loss_data = torch.mean((data(model, x_grid, t_grid, u_data))**2)

        # === Total loss ===
        loss = (l_pde * loss_pde + 
                l_data * loss_data)


        # Backpropagation
        loss.backward()
        optimizer.step()

        # Store history
        history.append([loss.item(), l_pde * loss_pde.item(), l_data * loss_data.item()])

        if epoch % print_every == 0:
            print(f"[{epoch}] Total: {history[epoch][0]:.4e} PDE: {history[epoch][1]:.4e} Data: {history[epoch][2].item():.4e}")

    return history



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



