import matplotlib.pyplot as plt
from pylab import contour, cm
import numpy as np

def plot_it(func, x_star=[0,0], x_0=None, search_line=None, bound_line=None, **params):
    figsize = params.get('figsize', (10,10))
    n_steps = 100
    step = 0.1

    space_x = np.linspace(x_star[0] - n_steps * step, x_star[0] +  n_steps * step, n_steps)
    space_y = np.linspace(x_star[1] - n_steps * step, x_star[1] + n_steps * step, n_steps)
    f = np.vectorize(lambda x, y: func([x, y]))
    X, Y = np.meshgrid(space_x, space_y)
    Z = f(X, Y)
    
    f, ax = plt.subplots(1, 1, figsize=figsize)
    
    levels = func(x_star) + np.linspace(0, 14, 14)**2
    
    ax.plot(*x_star, 'b*')
    
    if np.any(x_0) != None:
        ax.plot(*x_0, 'go')
        
    ax.contour(X, Y, Z, levels=levels, cmap=cm.GnBu, extent=[0, 1, 0, 1])
    
    if np.any(bound_line) != None:
        ax.plot(*bound_line, 'r')
        
    if np.any(search_line) != None: 
        ax.plot(search_line[:, 0], search_line[:, 1])
        
    return f

def plot_surf(func, x_star, **params):
    n_steps = params.get('n_steps', 100)
    step = params.get('step', 0.1)

    space_x = np.linspace(x_star[0] - n_steps * step, x_star[0] +  n_steps * step, n_steps)
    space_y = np.linspace(x_star[1] - n_steps * step, x_star[1] + n_steps * step, n_steps)
    f = np.vectorize(lambda x, y: func([x, y]))
    X, Y = np.meshgrid(space_x, space_y)
    Z = f(X, Y)
    
    f, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    bound = params.get('bounds', 14)
    levels = func(x_star) + np.linspace(-bound, bound, params.get('n_lvls', 14))
    
    ax.plot(*x_star, 'b*')
    
    ax.contour(X, Y, Z, levels=levels, cmap=cm.GnBu, extent=[0, 1, 0, 1])
    
    return f

def approx_it(y):
    n = len(y)
    x = np.arange(1, n+1)
    norm = np.linalg.norm
    E = np.sum
    

    coef = 1 / (n * norm(x)**2 - E(x)**2)
    A = np.array([[n, -E(x)], 
                  [-E(x), norm(x)**2]])
    b = np.array([E(x*y), E(y)])
    
    return coef * np.dot(A, b)

def plot_convergence(fs, optimum):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (13,4))
    xs = np.arange(1, len(fs)+1)
    ax1.plot(xs, (fs - optimum), 'x')
    ax2.plot(xs, np.log(fs - optimum), 'x')
    
    [a, b] = approx_it(np.log(fs - optimum))

    approx = np.vectorize(lambda k: np.exp(a * k  + b))
    ks = np.linspace(1, len(fs)+1, len(fs)*2)
    ax1.plot(ks, approx(ks), ':')
    ax2.plot(xs, a * xs +  b, ':')

    return fig
    