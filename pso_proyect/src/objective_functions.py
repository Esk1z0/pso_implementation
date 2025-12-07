import numpy as np

def rastrigin (x) :
    assert all(abs(xi)<=5.12 for xi in x)
    A = 10
    return A * len(x) + sum([ (xi**2 - A * np.cos(2*np.pi*xi)) for xi in x ])


    