# AI Engineering from Scratch: companion code

One small repo that grows every episode, from a single neuron (Ep 1) to a tiny GPT (Ep 5).
Plain Python first, so every line is readable; NumPy and MLX arrive when speed starts to matter.

## Ep 1: How a Machine Learns
| file | what it is |
|---|---|
| `micro.py` | a tiny autograd engine (`Value`) plus `Neuron`, `Layer`, `MLP`, adapted from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) |
| `coffee.py` | a **pretend** coffee cart: made-up temperatures and iced coffees sold (synthetic data, recipe included) |
| `ep01_coffee.py` | train one neuron on the cart, with real gradient descent, and watch the loss fall |
| `ep01_export.py` | every experiment from the video (learning rates, the loss landscape, the heat-wave curve, overfitting), saved to `runs/ep01.json` |

```bash
python3 ep01_coffee.py        # learning rate 0.1
python3 ep01_coffee.py 1.0    # homework: what happens?
python3 ep01_export.py        # all the runs in the video (~20 s)
```

No installs needed: Python 3.9+ standard library only.

## Credit
`micro.py` is adapted from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), MIT License, Copyright (c) 2020 Andrej Karpathy. Its full license notice is in [`LICENSE-micrograd`](LICENSE-micrograd).
