import torch
import numpy as np

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


######### device inside !!!
def accuracy(model, dataset_loader, device, solver = None, solver_options = None):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        
        if solver is not None:
            out = model(x, solver, solver_options).cpu().detach().numpy()
        else:
            out = model(x).cpu().detach().numpy()

        predicted_class = np.argmax(out, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def sn_test(model, test_loader, device, solvers, solver_options, nsteps_grid):
    model.eval()
    for solver in solvers:
        solver.freeze_params()
    
    accs = []
    for nsteps in nsteps_grid:
        for solver in solvers:
            solver.grid_constructor =  lambda t: torch.linspace(t[0], t[-1], nsteps + 1)

        with torch.no_grad():
            acc = accuracy(model, test_loader, device, solvers, solver_options)
            accs.append(acc)
            
    return accs