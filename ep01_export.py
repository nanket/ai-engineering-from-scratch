"""Run every Ep 1 experiment for real and save it to runs/ep01.json, so the video plays back real numbers.

    python3 ep01_export.py
"""
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

from coffee import (BENT_CUPS, BENT_TEMPS, CUPS, TEMPS, TEST_CUPS, TEST_TEMPS, TRAIN_CUPS, TRAIN_TEMPS, scale)
from micro import MLP, Neuron, Value

HERE = Path(__file__).parent
R = lambda v, n=4: round(v, n)
GRID_T = list(range(8, 43))            # temperatures where curves are sampled for drawing


def mse(model, temps, cups):
    return sum((model([scale(t)]) - y) ** 2 for t, y in zip(temps, cups)) * (1 / len(cups))


def train(model, temps, cups, lr, steps, every=1, test=None, snaps=()):
    """Plain gradient descent on the average squared error. Logs real numbers as it goes."""
    log, curves, t0 = [], {}, time.perf_counter()
    for step in range(steps + 1):
        loss = mse(model, temps, cups)
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        if step % every == 0 or step == steps:
            row = {"step": step, "loss": R(loss.data)}
            if isinstance(model, Neuron):
                row.update(w=R(model.w[0].data), b=R(model.b.data), dw=R(model.w[0].grad), db=R(model.b.grad))
            if test:
                row["test"] = R(mse(model, *test).data)
            log.append(row)
        if step in snaps:
            curves[step] = [R(model([scale(t)]).data, 2) for t in GRID_T]
        if step < steps:
            for p in model.parameters():
                p.data -= lr * p.grad
    secs = time.perf_counter() - t0
    return log, curves, secs


def line_run(lr, epochs, seed=85):
    random.seed(seed)
    n = Neuron(1, relu=False)
    log, _, secs = train(n, TEMPS, CUPS, lr, epochs)
    return {"lr": lr, "epochs": epochs, "log": log, "seconds": R(secs, 5)}


def hand_step(seed=85, temp=35, cups=48, lr=0.1):
    """One day, one step, worked out by the autograd engine (the video does the same math by hand)."""
    random.seed(seed)
    n = Neuron(1, relu=False)
    w, b, x = n.w[0], n.b, scale(temp)
    wx = w * x
    pred = wx + b
    err = pred - cups
    loss = err ** 2
    loss.backward()
    nodes = {k: {"data": R(v.data), "grad": R(v.grad)} for k, v in
             dict(w=w, b=b, wx=wx, pred=pred, err=err, loss=loss).items()}
    w2, b2 = w.data - lr * w.grad, b.data - lr * b.grad
    pred2 = w2 * x + b2
    return {"temp": temp, "x": x, "cups": cups, "lr": lr, "nodes": nodes,
            "after": {"w": R(w2), "b": R(b2), "pred": R(pred2), "err": R(pred2 - cups), "loss": R((pred2 - cups) ** 2)}}


def surface(w_range=(-15, 45), b_range=(-15, 60), n=61):
    xs = [scale(t) for t in TEMPS]
    ws = [w_range[0] + (w_range[1] - w_range[0]) * i / (n - 1) for i in range(n)]
    bs = [b_range[0] + (b_range[1] - b_range[0]) * j / (n - 1) for j in range(n)]
    loss = [[R(sum((w * x + b - y) ** 2 for x, y in zip(xs, CUPS)) / len(xs), 2) for w in ws] for b in bs]
    return {"w": [R(v, 3) for v in ws], "b": [R(v, 3) for v in bs], "loss": loss, "rows": "b", "cols": "w"}


def best_line():
    xs = [scale(t) for t in TEMPS]
    n, sx, sy = len(xs), sum(xs), sum(CUPS)
    sxx, sxy = sum(x * x for x in xs), sum(x * y for x, y in zip(xs, CUPS))
    w = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b = (sy - w * sx) / n
    return {"w": R(w), "b": R(b), "loss": R(sum((w * x + b - y) ** 2 for x, y in zip(xs, CUPS)) / n)}


def mlp_params(m):
    hid, out = m.layers[0].neurons, m.layers[1].neurons[0]
    return {"hidden": [{"w": R(h.w[0].data), "b": R(h.b.data), "v": R(out.w[i].data)} for i, h in enumerate(hid)],
            "c": R(out.b.data)}


def bent(seed=4, hidden=8, lr=0.004, steps=3000):
    out = {"temps": BENT_TEMPS, "cups": BENT_CUPS}
    random.seed(85)
    line = Neuron(1, relu=False)
    log, _, secs = train(line, BENT_TEMPS, BENT_CUPS, 0.1, 100, every=10)
    out["line"] = {"w": R(line.w[0].data), "b": R(line.b.data), "loss": log[-1]["loss"], "seconds": R(secs, 4)}
    # the same network with the bends removed: two straight layers fold into one line
    random.seed(seed)
    flat = MLP(1, [hidden, 1])
    for neuron in flat.layers[0].neurons:
        neuron.use_relu = False
    log, curves, secs = train(flat, BENT_TEMPS, BENT_CUPS, lr, steps, every=100, snaps=(steps,))
    out["no_relu"] = {"loss": log[-1]["loss"], "curve": curves[steps], "params": mlp_params(flat), "seconds": R(secs, 3)}
    random.seed(seed)
    m = MLP(1, [hidden, 1])
    snaps = (0, 25, 50, 100, 200, 400, 800, 1600, steps)
    log, curves, secs = train(m, BENT_TEMPS, BENT_CUPS, lr, steps, every=10, snaps=snaps)
    out["mlp"] = {"sizes": [1, hidden, 1], "n_params": len(m.parameters()), "lr": lr, "steps": steps, "seed": seed,
                  "log": [[r["step"], r["loss"]] for r in log], "curves": {str(k): v for k, v in curves.items()},
                  "params": mlp_params(m), "seconds": R(secs, 3)}
    return out


def overfit(seed=1, sizes=(32, 1), lr=0.004, steps=4000):
    random.seed(seed)
    m = MLP(1, list(sizes))
    snaps = (0, 200, 400, 620, 800, 1200, 1600, 2400, 3200, steps)
    log, curves, secs = train(m, TRAIN_TEMPS, TRAIN_CUPS, lr, steps, every=20, test=(TEST_TEMPS, TEST_CUPS), snaps=snaps)
    best = min(log[1:], key=lambda r: r["test"])
    return {"train": [TRAIN_TEMPS, TRAIN_CUPS], "test": [TEST_TEMPS, TEST_CUPS], "sizes": [1] + list(sizes),
            "n_params": len(m.parameters()), "lr": lr, "steps": steps, "seed": seed,
            "log": [[r["step"], r["loss"], r["test"]] for r in log], "best_test": best,
            "curves": {str(k): v for k, v in curves.items()}, "seconds": R(secs, 3)}


if __name__ == "__main__":
    t0 = time.perf_counter()
    run = {"made_on": {"machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor(),
                       "python": platform.python_version(), "os": platform.platform()},
           "temps": TEMPS, "cups": CUPS, "grid_temps": GRID_T}
    run["best_line"] = best_line()
    run["main"] = line_run(0.1, 60)
    fin = run["main"]["log"][-1]
    run["main"]["predictions"] = [R(fin["w"] * scale(t) + fin["b"], 2) for t in TEMPS]
    run["lr_runs"] = {str(lr): line_run(lr, 60) for lr in (0.01, 0.1, 0.7, 0.9, 1.0)}
    run["hand_step"] = hand_step()
    run["surface"] = surface()
    print("line runs done", flush=True)
    run["bent"] = bent()
    print("bent done", flush=True)
    run["overfit"] = overfit()
    print("overfit done", flush=True)
    # the exact terminal output the video shows in the "build it" segment
    out = subprocess.run([sys.executable, "ep01_coffee.py"], cwd=HERE, capture_output=True, text=True).stdout
    run["terminal"] = out.splitlines()
    run["export_seconds"] = R(time.perf_counter() - t0, 2)
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "ep01.json").write_text(json.dumps(run))
    print(f"runs/ep01.json written in {run['export_seconds']} s")
