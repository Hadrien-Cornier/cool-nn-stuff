from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('MacOSX')  # Replace 'TkAgg' with your preferred backend

# The goal is to express all polynomials with a neural network that only uses relu activation functions
# many thanks to Nikolaj-K @ https://www.youtube.com/watch?v=PApGm1TKFHQ for the topic

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
    return 0.5 * (relu(x) - 2 * relu(x - 0.5))


def plot(x: np.ndarray, y: np.ndarray, title: str = "", color: str = "blue"):
    plt.plot(x, y, label=title, color=color)


def apply_n(fn: Callable[[np.ndarray], np.ndarray], n: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: apply_n(fn, n - 1)(fn(x)) if n > 0 else x


def double_triangle(x: np.ndarray) -> np.ndarray:
    # obtained through trial and error
    # creates two peaks
    return triangle_from_2_relus(2 * triangle_from_2_relus(2 * x)) + triangle_from_2_relus(
        2 * triangle_from_2_relus(2 * (x - 0.5)))


def square_fractal(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    # obtained through double_triangle
    return lambda x: fn(2 * fn(2 * x)) + fn(2 * fn(2 * (x - 0.5)))


if __name__ == "__main__":

    for N in range(1, 4):
        X = np.linspace(0.0, 1.0, 10000)
        Y = apply_n(square_fractal, N)(triangle_from_2_relus)(X)
        plot(X, Y, f"order {N}", color="blue")

    plt.legend(loc='best')
    plt.title("repeated relus")
    plt.show()

# plot both of those on the same graph with different colors
