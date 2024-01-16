from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('MacOSX')  # Replace 'TkAgg' with your preferred backend

# The goal is to express all polynomials with a neural network that only uses relu activation functions
# many thanks to Nikolaj-K @ https://www.youtube.com/watch?v=PApGm1TKFHQ for the topic
# rewrote it in a different way

def R(x: np.ndarray) -> np.ndarray:
    """
    Relu function
    :param x: 1D array in R^n
    :return: Relu(x) 1D array in R^n
    """
    return np.maximum(x, 0)


def T(x: np.ndarray) -> np.ndarray:
    """
    Triangle function
    """
    return -R(x) - R(-x) + 1. / 2


def M(t: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Minification function (takes a triangle T and turns it into a double triangle M(T))
    :param t: the triangle function
    :return:
    """
    return lambda x: t(2 * x + 1) + t(2 * x - 1)


def plot(X: np.ndarray, Y: np.ndarray, label: str, color: str) -> None:
    """
    Plot a function
    :param X: x values
    :param Y: y values
    :param label: label for the plot
    :param color: color for the plot
    :return: None
    """
    plt.plot(X, Y, label=label, color=color)


def set_grid() -> None:
    """
    Set the grid for the plot
    :return: None
    """
    plt.grid(True)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)


if __name__ == "__main__":
    # Our map is the square of all x,y s.t. x in [-0.5,0.5] and y in [0.0,0.5]
    # This is shifted from Nikolaj-K's video where he uses [0,1]x[0,1]
    # imo this makes it easier to understand

    # We want to express the square function in terms of relus and linear combinations
    # essentiually we want to see -x**2+0.5
    color_cycle = matplotlib.colormaps["viridis"]
    X = np.linspace(-0.5, 0.5, 10000)
    Y = T(X)
    assert Y.max() <= 0.5 and Y.min() >= -0.5
    plot(X, T(X), "$T_1$", color=color_cycle(0.0))
    plot(X, T(2 * X + 1), "$T(2 * X + 1)$", color=color_cycle(0.25))
    plot(X, T(2 * X - 1), "$T(2 * X - 1)$", color=color_cycle(0.5))
    plot(X, -T(2 * X - 1), "$T(2 * X - 1)$", color=color_cycle(0.75))
    plot(X, -T(-2 * X - 1), "$T(2 * X - 1)$", color=color_cycle(1.0))
    plot(X, -2*X**2 + 0.5, "$-2*X^2+1/2$", color=color_cycle(0.9))

    # print()
    # plot(X, M(T)(X), "$T_2$", color=color_cycle(0.5))
    set_grid()
    plt.legend(loc='best')
    plt.title("getting $-2*X^2+1/2$ from Relus is the goal")
    plt.show()

# plot both of those on the same graph with different colors
