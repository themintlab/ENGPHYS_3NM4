# Polyfit Model UI and Plot
import numpy as np
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
from math import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ipywidgets import VBox, HBox, Button, Text, Label, Output
from IPython.display import display, clear_output
np.random.seed(42)
from ipywidgets import ToggleButtons
X = np.linspace(0, 3, 50).reshape(1, -1)

def init_weights():
    global W1, b1, W2, b2
    W1 = np.random.randn(4,1) * 0.5
    b1 = np.random.randn(4,1) * 0.5
    W2 = np.random.randn(1,4) * 0.5
    b2 = np.random.randn(1,1) * 0.5

init_weights()
loss_history = []

def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1 - np.tanh(x)**2

def forward(X):
    Z1 = W1 @ X + b1
    A1 = tanh(Z1)
    Z2 = W2 @ A1 + b2
    A2 = Z2
    return Z1, A1, Z2, A2

def compute_loss(A2, y): return np.mean((A2 - y)**2)

def backward(X, y, Z1, A1, A2, lr=0.1):
    global W1, b1, W2, b2
    m = X.shape[1]
    dZ2 = (A2 - y) / m
    dW2 = dZ2 @ A1.T
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * tanh_derivative(Z1)
    dW1 = dZ1 @ X.T
    db1 = np.sum(dZ1, axis=1, keepdims=True)
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

output_plot = widgets.Output()
output_table = widgets.Output()
status_label = widgets.Label(value="Press 'Step' to begin training.")

def plot_nn_diagram():
    layer_x = [0, 2, 4]
    y_positions = [[0], [-3, -1, 1, 3], [0]]
    annotations = []
    shapes = []
    layer_titles = ['x', 'Hidden Layer', 'f(x)']

    for l, ys in enumerate(y_positions):
        for i, y in enumerate(ys):
            shapes.append(dict(type="circle", xref="x", yref="y",
                x0=layer_x[l]-0.2, x1=layer_x[l]+0.2,
                y0=y-0.2, y1=y+0.2, line_color="black"))
            label = 'x' if l==0 else ('y' if l==2 else f'h{i+1}')
            annotations.append(dict(x=layer_x[l], y=y, text=label,
                showarrow=False, font=dict(size=12)))
            if l == 1:
                bias = b1[i, 0]
                annotations.append(dict(x=layer_x[l]+0.3, y=y, text=f"b={bias:.2f}",
                    showarrow=False, font=dict(size=10, color='gray')))
            elif l == 2:
                annotations.append(dict(x=layer_x[l]+0.3, y=y, text=f"b={b2[0,0]:.2f}",
                    showarrow=False, font=dict(size=10, color='gray')))

    for i, y_in in enumerate(y_positions[0]):
        for j, y_hid in enumerate(y_positions[1]):
            weight = W1[j, i]
            shapes.append(dict(type='line', xref='x', yref='y',
                x0=layer_x[0]+0.2, y0=y_in, x1=layer_x[1]-0.2, y1=y_hid,
                line=dict(color='blue')))
            annotations.append(dict(x=1, y=(y_in + y_hid)/2, text=f"{weight:.2f}",
                showarrow=False, font=dict(size=9, color='blue')))

    for j, y_hid in enumerate(y_positions[1]):
        for k, y_out in enumerate(y_positions[2]):
            weight = W2[k, j]
            shapes.append(dict(type='line', xref='x', yref='y',
                x0=layer_x[1]+0.2, y0=y_hid, x1=layer_x[2]-0.2, y1=y_out,
                line=dict(color='red')))
            annotations.append(dict(x=3, y=(y_hid + y_out)/2, text=f"{weight:.2f}",
                showarrow=False, font=dict(size=9, color='red')))

    for i, x in enumerate(layer_x):
        annotations.append(dict(x=x, y=max(y_positions[1])+1.5,
            text=layer_titles[i], showarrow=False, font=dict(size=14)))

    fig = go.Figure()
    fig.update_layout(shapes=shapes, annotations=annotations,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      title='Neural Network Structure', height=500, width=700,
                      margin=dict(l=20, r=20, t=40, b=20))
    fig.show()

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ipywidgets import VBox, HBox, Button, Text, Label, Output
from IPython.display import display, clear_output

depth = 1
width = 3
activation = np.tanh
activation_derivative = lambda x: 1 - np.tanh(x) ** 2

X = np.linspace(0, 3, 50).reshape(-1, 1)
X2 = np.linspace(0, 3.5, 50).reshape(-1, 1)
true_function = None
losses = []
weights, biases = [], []
weight_history, bias_history = [], []

function_input = Text(value='exp(-x/2)*sin(2*pi*x)', description='f(x):')
save_button = Button(description='Save Function', button_style='success')
reset_button = Button(description='Reset Model', button_style='warning')
status_label = Label()

step_label = Label(value="Step:")
step_1 = Button(description='1')
step_10 = Button(description='10')
step_100 = Button(description='100')
step_1000 = Button(description='1000')
step_10000 = Button(description='10000')

depth_label = Label(value="Change Depth (layers):")
depth_add = Button(description='+1')
depth_sub = Button(description='-1')
width_label = Label(value="Change Width (neurons):")
width_add = Button(description='+1')
width_sub = Button(description='-1')

output_plot = Output()
network_plot = Output(layout={"height": "500px", "overflow": "auto"})
metrics_plot = Output()

def init_model():
    global weights, biases, weight_history, bias_history
    layers = [1] + [width]*depth + [1]
    weights = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]) for i in range(len(layers)-1)]
    biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]
    weight_history = [[] for _ in weights]
    bias_history = [[] for _ in biases]

def forward_pass(x):
    activations = [x]
    zs = []
    a = x
    for w, b in zip(weights[:-1], biases[:-1]):
        z = a @ w + b
        zs.append(z)
        a = activation(z)
        activations.append(a)
    z = a @ weights[-1] + biases[-1]
    zs.append(z)
    activations.append(z)  # no activation on final output
    return zs, activations

def backward_pass(zs, activations, y_true, lr=0.01):
    global weights, biases, weight_history, bias_history
    grads_w = [None] * len(weights)
    grads_b = [None] * len(biases)

    delta = (activations[-1] - y_true)
    grads_w[-1] = activations[-2].T @ delta / len(X)
    grads_b[-1] = np.mean(delta, axis=0, keepdims=True)

    for l in range(2, len(weights)+1):
        z = zs[-l]
        sp = activation_derivative(z)
        delta = (delta @ weights[-l+1].T) * sp
        grads_w[-l] = activations[-l-1].T @ delta / len(X)
        grads_b[-l] = np.mean(delta, axis=0, keepdims=True)

    for i in range(len(weights)):
        weights[i] -= lr * grads_w[i]
        biases[i] -= lr * grads_b[i]
        weight_history[i].append(weights[i].copy())
        bias_history[i].append(biases[i].copy())

    return np.mean((activations[-1] - y_true)**2)

def step(n=1):
    global losses
    if true_function is None: return
    y_true = true_function(X)
    for _ in range(n):
        zs, activations = forward_pass(X)
        loss = backward_pass(zs, activations, y_true)
        losses.append(loss)
    update_plots()

def reset_model(_=None):
    init_model()
    losses.clear()
    status_label.value = "Model reset."
    update_plots()

def save_function(_=None):
    global true_function, losses
    try:
        code = function_input.value
        true_function = lambda x: eval(code, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "pi": np.pi})
        losses.clear()
        status_label.value = "Function saved."
        update_plots()
    except Exception as e:
        status_label.value = f"Error: {e}"

def change_depth(d):
    global depth
    depth = max(0, depth + d)
    reset_model()

def change_width(d):
    global width
    width = max(1, width + d)
    reset_model()

save_button.on_click(save_function)
reset_button.on_click(reset_model)
step_1.on_click(lambda b: step(1))
step_10.on_click(lambda b: step(10))
step_100.on_click(lambda b: step(100))
step_1000.on_click(lambda b: step(1000))
step_10000.on_click(lambda b: step(10000))
depth_add.on_click(lambda b: change_depth(1))
depth_sub.on_click(lambda b: change_depth(-1))
width_add.on_click(lambda b: change_width(1))
width_sub.on_click(lambda b: change_width(-1))

def draw_network(activations):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    G = nx.DiGraph()
    labels = {}
    pos = {}
    edge_labels = {}
    edge_colors = {}
    layer_sizes = [1] + [width] * depth + [1]

    max_layer_size = max(layer_sizes)

    for l, size in enumerate(layer_sizes):
        layer_offset = (max_layer_size - size) / 2
        for n in range(size):
            node = f"L{l}N{n}"
            pos[node] = (l, -(n + layer_offset))
            G.add_node(node)

            # Label nodes
            if l == 0:
                labels[node] = "x"
            elif l == len(layer_sizes) - 1:
                labels[node] = "f(x)"
            else:
                try:
                    act_val = activations[l][0, n]
                    bias_val = biases[l - 1][0, n]
                    labels[node] = f"{act_val:.2f}\nb={bias_val:.2f}"
                except Exception:
                    labels[node] = f"H{l-1}N{n}"

            # Add edges
            if l > 0:
                for p in range(layer_sizes[l - 1]):
                    prev_node = f"L{l-1}N{p}"
                    G.add_edge(prev_node, node)
                    w_val = weights[l - 1][p, n]
                    edge_labels[(prev_node, node)] = f"{w_val:.2f}"
                    norm_val = np.tanh(w_val)  # normalized for color
                    edge_colors[(prev_node, node)] = plt.cm.bwr((norm_val + 1) / 2)

    return G, pos, labels, edge_labels, edge_colors


def update_plots():
    output_plot.clear_output(wait=True)
    metrics_plot.clear_output(wait=True)
    network_plot.clear_output(wait=True)
    with output_plot:
        if true_function:
            plt.figure(figsize=(6, 3))
            y_true = true_function(X)
            _, activations = forward_pass(X2)
            y_pred = activations[-1]
            plt.plot(X, y_true, label='True')
            plt.scatter(X2, y_pred, label='NN')
            plt.legend()
            plt.title("Function vs NN Output")
            plt.grid(True)
            plt.show()

    with metrics_plot:
        if losses:
            plt.figure(figsize=(6, 3))
            plt.plot(losses, label="Loss", color='red')
            for i, history in enumerate(weight_history):
                flat_vals = [w.flatten()[0] for w in history]
                plt.plot(flat_vals, label=f"W{i}_0", alpha=0.5)
            for i, history in enumerate(bias_history):
                flat_vals = [b.flatten()[0] for b in history]
                plt.plot(flat_vals, label=f"b{i}_0", linestyle='dotted', alpha=0.5)
            plt.title("Loss and Parameter Changes")

            plt.grid(True)
            plt.legend()
            plt.show()
    with network_plot:
        if true_function:
            activations = forward_pass(X)  # Ensure activations are defined
            G, pos, labels, edge_labels, edge_colors = draw_network(activations)
            edge_color_vals = [edge_colors.get(edge, '#888888') for edge in G.edges()]
            nx.draw(G, pos, labels=labels, node_color='lightblue', node_size=600,
                    edge_color=edge_color_vals, edge_cmap=plt.cm.bwr, arrows=True)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            plt.title("Neural Network Diagram")
            plt.show()

top_controls = HBox([function_input, save_button, reset_button])
step_controls = HBox([step_label, step_1, step_10, step_100, step_1000, step_10000])
depth_controls = HBox([depth_label, depth_sub, depth_add])
width_controls = HBox([width_label, width_sub, width_add])
graph_row = HBox([output_plot, metrics_plot])
full_ui = VBox([top_controls, status_label, step_controls, depth_controls, width_controls, network_plot, graph_row])
display(full_ui)

init_model()
update_plots()
