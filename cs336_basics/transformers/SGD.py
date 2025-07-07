from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import matplotlib.pyplot as plt

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

def run_sgd_experiment(lr, steps=100):
    print(f"--- Running SGD with lr={lr} ---")
    torch.manual_seed(0)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    losses = []
    for t in range(steps):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.cpu().item())
        if t < 10: # Print first 10 steps
             print(f"Step {t}, Loss: {loss.cpu().item()}")
        loss.backward()
        opt.step()
    print("\n")
    return losses

def main():
    learning_rates = [1, 1e1, 1e2, 1e3]
    num_steps = 100

    for lr in learning_rates:
        losses = run_sgd_experiment(lr, steps=num_steps)
        plt.figure()
        plt.plot(range(num_steps), losses)
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title(f"SGD Loss Curve (lr={lr})")
        plot_filename = f"sgd_lr_{lr}_loss_curve.png"
        plt.savefig(plot_filename)
        print(f"Saved plot to {plot_filename}")

if __name__ == "__main__":
    main()