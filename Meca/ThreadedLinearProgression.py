import numpy as np
import threading as thr
import multiprocessing as mp

b = 10

def get_variables_w(stack: np.array) -> list:
    w = 10
    wstack = [w for _ in range(len(stack))]
    return wstack

def f(stack: np.array, wstack: list, b: float) -> np.array:
    suma = b
    for i in range(len(stack)):
        suma += wstack[i] * stack[i]
    return suma

def J(stack: np.array, wstack: list, y: np.array, b: float) -> float:
    return ((f(stack, wstack, b) - y)**2).mean()

def dJ_dwn_dJ_db(stack: np.array, wstack: list, y: np.array, b: float) -> list:
    dJ_dwn_dJ_dblist = []
    error = f(stack, wstack, b) - y

    def derivada_parcial(i: int) -> float:
        dJ_dwn_dJ_dblist.append(((2 * error * stack[i]).mean()))
    
    threads = []
    for i in range(len(wstack)):
        thread = thr.Thread(target= derivada_parcial, args= (i, ))
        threads.append(thread)
        thread.start()
    

    for thread in threads:
        thread.join()

    dJ_dwn_dJ_dblist.append((2 * error).mean())
    return dJ_dwn_dJ_dblist

def descenso_de_gradiente_eq(stack: np.array, wstack: list, y: np.array, alpha: float, b: float) -> tuple:
    derivatives = dJ_dwn_dJ_db(stack, wstack, y, b)
    for i in range(len(wstack)):
        wstack[i] -= alpha * derivatives[i]
    b -= alpha * derivatives[-1]
    return wstack, b

def descenso_de_gradiente(stack: np.array, wstack: list, y: np.array, alpha: float, b: float, lock, results) -> None:
    ws, b = descenso_de_gradiente_eq(stack, wstack, y, alpha, b)
    with lock:
        results.append((ws, b))

def training(stack: np.array, y: np.array, alpha: float, epochs: int) -> tuple:
    ws = get_variables_w(stack)
    global b

    manager = mp.Manager()
    results = manager.list()
    lock = manager.Lock()

    for i in range(epochs):
        processes = []
        process = mp.Process(target= descenso_de_gradiente, args= (stack, ws, y, alpha, b, lock, results))
        processes.append(process)
        process.start()
        
        for process in processes:
            process.join()

        if len(results) == 0:
            raise RuntimeError("Results list is empty. No results were appended. ")
        ws, b = results[-1]

        if (i+1) % 100 == 0:
            print('El costo es: ', J(stack, ws, y, b))


def model(weights: list, b: float) -> None:
    output = 'El modelo f es: '
    for i in range(len(weights)):
        if i == 0:
            output += f'{weights[i]}x0'
        else:
            output += f' + {weights[i]}x{i}'
    output += f' + {b}'
    print(output)


if __name__ == '__main__':
    x1: np.array = np.array([0, 1, 2, 3, 4])
    x2: np.array = np.array([0, 1, 2, 3, 4])
    y: np.array = np.array([3, 7, 11, 15, 19])

    epochs: int = 100000
    alpha: float = 1e-4
    
    stack = np.stack([x1, x2])

    ws, b = training(stack, y, alpha, epochs)
    model(ws, b)

