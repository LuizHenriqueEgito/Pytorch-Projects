import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
from typing import List

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

# TODO: crie uma função de train para treinar apenas com as duas features selecionadas 
@torch.no_grad()
def plot_decision_boundary(model, dataset, color_map, s: int, cols: List[int], steps: int = 100):
    xmin, xmax = dataset.tensors[0][:, cols[0]].min().item() - 1, dataset.tensors[0][:, cols[0]].max().item() + 1
    ymin, ymax = dataset.tensors[0][:, cols[1]].min().item() - 1, dataset.tensors[0][:, cols[1]].max().item() + 1
    
    steps = steps
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)
    data_plot = Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
    
    model.eval()
    labels_predicted = model(data_plot)
    labels_predicted = [0 if value <= 0.5 else 1 for value in labels_predicted.detach().numpy()]
    z = np.array(labels_predicted).reshape(xx.shape)
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=color_map, alpha=0.5)
    ax.scatter(
        dataset.tensors[0][:, cols[0]],
        dataset.tensors[0][:, 1],
        c=dataset.tensors[1],
        cmap=color_map,
        edgecolor='white',
        s=s
    )
    plt.show()
    return fig, ax