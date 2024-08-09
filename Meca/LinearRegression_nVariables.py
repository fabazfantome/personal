import numpy as np


w1: float = 10
w2: float = 10
b: float = 10

def f(x1: np.array, x2: np.array) -> np.array:
    global w1, w2, b 
    return w1*x1 + w2*x2 + b

def J(x1: np.array, x2: np.array, y: np.array) -> float:
    return ((f(x1, x2) - y)**2).mean() #Resultó que utilizando .mean el código era más preciso(?)

def dJ_dw1(x1: np.array, x2: np.array, y: np.array) -> float:
    global w1, w2, b
    return (2*(f(x1, x2) - y)*x1).mean()

def dJ_dw2(x1: np.array, x2: np.array, y: np.array) -> float:
    global w1, w2, b
    return (2*(f(x1, x2) - y)*x2).mean()

def dJ_db(x1: np.array, x2: np.array, y: np.array) -> float:
    global w1, w2, b
    return (2*(f(x1, x2) - y)).mean()

def descenso_de_gradiente(x1: np.array, x2: np.array, y: np.array, alpha: float) -> None:
    global w1, w2, b
    w1 -= alpha*dJ_dw1(x1, x2, y)
    w2 -= alpha*dJ_dw2(x1, x2, y)
    b -= alpha*dJ_db(x1, x2, y)

def training(x1: np.array, x2: np.array, y: np.array, epochs: int, alpha: float) -> None:
    for _ in range(epochs):
        descenso_de_gradiente(x1, x2, y, alpha)
        if epochs % 100 == 0:
            print('El costo es: ', J(x1, x2, y))


if __name__ == '__main__':
    x1: np.array = np.array([0, 1, 2, 3, 4])
    x2: np.array = np.array([0, 1, 2, 3, 4])
    y: np.array = np.array([3, 7, 11, 15, 19])

    epochs: int = 100000
    alpha: float = 1e-4

    training(x1, x2, y, epochs, alpha)

    print(f'El modelo f es: {w1}x1 + {w2}x2 + {b}')

