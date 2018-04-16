from shapely.geometry import LineString, Point
import numpy as np

def exactline(v_k, Q):
    return np.linalg.norm(v_k) ** 2 /(np.dot(np.dot(v_k, Q), v_k))

def grad_desc(x0, f, f_grad, Q, **params):
    x = x0.copy()
    x_array = [x]
    f_array = [f(x)]
    
    iters = params.get('iters', None) or 20
   
    for k in range(iters):
        grad = f_grad(x)
        
        gamma = exactline(grad, Q)
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
                
        