import numpy as np
import torch

from neuralop.data.datasets.tensor_dataset import TensorDataset

def input_func(t, A, omega):
    return A*torch.sin(omega*t)

def solution_func(t, A, omega):
    return -A*omega*torch.cos(omega*t)

if __name__ == '__main__':

    res = 128 # Resolution
    n_A = 100
    n_omega = 100
    ts = torch.linspace(0, 4*np.pi, res)
    As = torch.rand(n_A)*5
    omegas = torch.rand(n_omega)*5
    As_test = torch.rand(n_A)*5
    omegas_test = torch.rand(n_omega)*5

    X_train = torch.zeros((n_A*n_omega, 1, res))
    y_train = torch.zeros((n_A*n_omega, 1, res))
    X_test = torch.zeros((n_A*n_omega, 1, res))
    y_test = torch.zeros((n_A*n_omega, 1, res))
    for i, A in enumerate(As):
        for j, omega in enumerate(omegas):
            X_train[j+n_omega*i] = input_func(ts, A, omega)
            y_train[j+n_omega*i] = solution_func(ts, A, omega)

            X_test[j+n_omega*i] = input_func(ts, As_test[i], omegas_test[j])
            y_test[j+n_omega*i] = solution_func(ts, As_test[i], omegas_test[j])

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    torch.save(train_dataset, './no_data/train_sine.pt')
    torch.save(test_dataset, './no_data/test_sine.pt')
