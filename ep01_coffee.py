"""Ep 1: teach ONE neuron to predict iced coffees from temperature.

    python3 ep01_coffee.py          # learning rate 0.1
    python3 ep01_coffee.py 1.0      # homework: what happens?
"""
import random
import sys
import time

from coffee import CUPS, TEMPS, scale
from micro import Neuron

random.seed(85)                         # the neuron is born random: w = -0.60, b = 0
neuron = Neuron(1, relu=False)          # prediction = w * temperature + b
xs = [scale(t) for t in TEMPS]          # 30 °C -> 1.0, 10 °C -> -1.0
ys = CUPS
lr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1   # learning rate: how big each step is

start = time.perf_counter()
for epoch in range(60):
    preds = [neuron([x]) for x in xs]                                   # 1. forward: guess every day
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) * (1 / len(ys))  # 2. loss: average squared miss
    for p in neuron.parameters():
        p.grad = 0.0
    loss.backward()                                                     # 3. backward: who is to blame?
    for p in neuron.parameters():
        p.data -= lr * p.grad                                           # 4. update: step against the blame
    print(f"epoch {epoch:2d}   loss {loss.data:10.2f}   w {neuron.w[0].data:7.2f}   b {neuron.b.data:7.2f}")

print(f"trained in {time.perf_counter() - start:.3f} s")
w, b = neuron.w[0].data, neuron.b.data
print(f"a 30 °C day: about {w * scale(30) + b:.0f} iced coffees")
