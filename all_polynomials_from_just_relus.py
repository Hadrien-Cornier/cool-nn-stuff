from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('MacOSX')  # Replace 'TkAgg' with your preferred backend


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def triangle(x: np.ndarray) -> np.ndarray:
    return x - 2 * relu(x - 0.5)


def identity_from_relus(x: np.ndarray) -> np.ndarray:
    return relu(x) - relu(x - 1)


def triangle_from_3_relus(x: np.ndarray) -> np.ndarray:
    # used 3 relus => not optimal because it can be done in two relus
    return identity_from_relus(x) - 2 * relu(x - 0.5)


def triangle_from_2_relus(x: np.ndarray) -> np.ndarray:
    # optimal in terms of number of neurons necessary
    return relu(x) - 2 * relu(x - 0.5)


def plot(x: np.ndarray, y: np.ndarray, title: str = ""):
    plt.plot(x, y)
    plt.title(title)
    plt.show()


def apply_n(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, n: int) -> np.ndarray:
    y = x
    for i in range(n):
        y = fn(y)
    return y


if __name__ == "__main__":
    X = np.linspace(0.0, 1.0, 100)
    N = 4
    Y = apply_n(triangle_from_2_relus, X, N)
    plot(X, Y, f"triangle^{N}")
