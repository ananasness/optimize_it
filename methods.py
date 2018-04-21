from shapely.geometry import LineString, Point
import numpy as np

def exactline(v_k, Q):
    return np.linalg.norm(v_k) ** 2 /(np.dot(np.dot(v_k, Q), v_k))

def exactline_new(x, p, Q, b):
    return np.dot(p, np.dot(Q,x) + b) /(np.dot(np.dot(p, Q), p))

def grad_desc(x0, f, f_grad, Q, b, **params):
    x = x0.copy()
    x_array = [x]
    f_array = [f(x)]
    
    iters = params.get('iters', None) or 20
   
    for k in range(iters):
        grad = f_grad(x)
        
        gamma = exactline_new(x, grad, Q, b)
        x = x - gamma * grad
        
        x_array.append(x)
        f_array.append(f(x))

    return np.array(x_array), np.array(f_array)

def coord_desc(x0, f, f_grad, Q,  **params):
    x = x0.copy()
    x_array = [x]
    f_array = [ f(x) ]
    iters = params.get('iters', None) or 20

    for k in range(iters):
        
        grad = f_grad(x)
        coord = (k+1) % len(x0)
        p = np.zeros_like(x0)
        p[coord] = 1
        
        x = x - p * grad * exactline(p*grad, Q)
        
        x_array.append(x)
        f_array.append(f(x))
        
    return np.array(x_array), np.array(f_array)


def proj(x, line): 
    line = np.array(line)
    
    if np.cross(line[:, 1] - line[:, 0], line[:, 1] - x) > 0:
        return x
    
    p = Point(x)
    l = LineString([line[:, 0], line[:, 1]])
    return l.interpolate(l.project(p)).coords[0]
    
def proj_desc(x0, f, f_grad, line,  **params):
    x = x0.copy()
    x_array = [x]
    f_array = [ f(x) ]
    iters = params.get('iters', None) or 100
    
    for k in range(iters):
        
        grad = f_grad(x)
        gamma = 1 /(1+k)
        x = x - gamma * grad
        x = proj(x, line)
        
        x_array.append(x)
        f_array.append(f(x))
        
    return np.array(x_array), np.array(f_array)

def ang(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def proj_ball(x, d, r):
    return r * (x - d) / np.linalg.norm(x - d) + d

def proj_ball_desc(x0, f, f_grad, center, r, **params):
    x = x0.copy()
    x_array = [x]
    f_array = [ f(x) ]
    iters = params.get('iters', None) or 100
    
    for k in range(iters):
        
        grad = f_grad(x)
        gamma = 1 /(1+k)
        x = x - gamma * grad
        x = proj_ball(x, center, r)
        
        x_array.append(x)
        f_array.append(f(x))
        
    return np.array(x_array), np.array(f_array)

def heavy_ball(x0, f, f_grad, **params):
    x = np.array(x0.copy())
    prev_x = x.copy()
    x_array = [x]
    f_array = [ f(x) ]
    iters = params.get('iters', None) or 100
    alpha = params.get('alpha', None) or 0.2
    beta = params.get('beta', None) or 0.65
    
    for k in range(iters):
        p = np.zeros_like(x0)
        coord = k % len(x0)
        p[coord] = 1
        
        grad = f_grad(x)
        
        new_x = x + alpha * (-grad) + beta * (x - prev_x)
        prev_x = x
        x = new_x
        
        x_array.append(x)
        f_array.append(f(x))
    return np.array(x_array), np.array(f_array)


def backtrack(x, f, f_grad, alpha, beta):
    grad = f_grad(x)
    gamma = 0.5
    delta_x = gamma * (-grad)
    while f(x + delta_x) > f(x) + alpha * gamma* np.dot(grad, delta_x):
        gamma = beta * gamma
    
    return gamma

def back_descent(x0, f, f_grad, **params):
    x = np.array(x0).copy()
    x_array = [x]
    f_array = [ f(x) ]
    iters = params.get('iters', None) or 100
    alpha = params.get('alpha', None) or 0.5
    beta = params.get('beta', None) or 0.5
    
    for k in range(iters):
        p = np.zeros_like(x0)
        coord = k % len(x0)
        p[coord] = 1
        
        grad = f_grad(x)
        
        gamma = backtrack(x, f, f_grad, alpha, beta)
        delta_x = gamma * (-grad)
        
        x = x + delta_x
        
        x_array.append(x)
        f_array.append(f(x))
    return np.array(x_array), np.array(f_array)
        
                
        