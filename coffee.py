"""A PRETEND coffee cart. Every number here is made up (synthetic), so we know the truth behind it.

Each day: the afternoon temperature (°C) and how many iced coffees the cart sold.
Generated once with random noise (see make_data() below) and pasted in, so you can read the data.
"""
import math
import random

# 20 days. Warmer days sell more iced coffee (with day-to-day noise).
TEMPS = [10, 11, 13, 14, 17, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35]
CUPS = [12, 7, 14, 14, 20, 32, 30, 24, 33, 31, 35, 35, 33, 36, 33, 42, 40, 41, 51, 48]

# A heat wave summer: past about 31 °C people stay inside, and sales DROP. No straight line fits this.
BENT_TEMPS = [11, 12, 13, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 33, 34, 35, 36, 37, 38, 40]
BENT_CUPS = [15, 14, 18, 15, 18, 22, 24, 26, 29, 33, 33, 34, 31, 34, 33, 36, 40, 36, 32, 30, 26, 21, 20, 15]

# Only 8 days of records to learn from (train), and 23 new days to check on (test). Same heat wave rule, noisier.
TRAIN_TEMPS = [11, 15, 19, 23, 27, 31, 35, 39]
TRAIN_CUPS = [11, 25, 20, 21, 37, 40, 32, 22]
TEST_TEMPS = [10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40]
TEST_CUPS = [10, 10, 11, 15, 18, 26, 26, 19, 30, 28, 37, 41, 37, 28, 41, 30, 44, 31, 31, 37, 16, 23, 12]


def scale(temp):
    """Neurons like small numbers near zero: count from a mild 20 °C day, in tens of degrees (30 °C -> 1.0)."""
    return (temp - 20) / 10


def make_data():
    """How the numbers above were made (for the curious)."""
    def softplus(z, k=0.12):
        return k * math.log1p(math.exp(z / k))
    def heat_wave(t):
        return 25 + 15 * scale(t) - 40 * softplus(scale(t) - 1.05)
    r = random.Random(2026)
    temps = sorted(r.sample(range(10, 36), 20))
    cups = [round(25 + 15 * scale(t) + r.gauss(0, 3)) for t in temps]
    r = random.Random(7)
    bent = sorted(set(r.sample(range(10, 41), 22)) | {38, 40})
    bent_cups = [round(heat_wave(t) + r.gauss(0, 2.5)) for t in bent]
    r = random.Random(13)
    train = [11, 15, 19, 23, 27, 31, 35, 39]
    train_cups = [round(heat_wave(t) + r.gauss(0, 5)) for t in train]
    test = [t for t in range(10, 41) if t not in train]
    test_cups = [round(heat_wave(t) + r.gauss(0, 5)) for t in test]
    return temps, cups, bent, bent_cups, train, train_cups, test, test_cups


if __name__ == "__main__":
    made = make_data()
    pasted = (TEMPS, CUPS, BENT_TEMPS, BENT_CUPS, TRAIN_TEMPS, TRAIN_CUPS, TEST_TEMPS, TEST_CUPS)
    print("data matches its recipe:", all(list(a) == list(b) for a, b in zip(made, pasted)))
