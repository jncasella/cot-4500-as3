import numpy as np

# Function for both Euler Method and Runge-Kutta method
def function(t, w):
    return t - (w ** 2)

def Euler(t, end_t, w, num_iter):
    h = (end_t - t) / num_iter
    
    for i in range(num_iter):        
        w += h * function(t, w)
        t += h
    
    return w

def RungeKutta(t, end_t, w, num_iter):
    k = np.zeros(4)
    h = (end_t - t) / num_iter
    
    for i in range(num_iter):
        k[0] = h * function(t, w)
        k[1] = h * function(t + (h / 2), w + (1/2) * k[0])
        k[2] = h * function(t + (h / 2), w + (1/2) * k[1])
        t += h
        k[3] = h * function(t, w + k[2])
        w += (1/6) * (k[0] + 2 * k[1] + 2 * k[2] + k[3])
    
    return w

def main():
    start_t = 0
    end_t = 2
    w = 1
    num_iter = 10
    print(Euler(start_t, end_t, w, num_iter), end = '\n\n')
    
    print(RungeKutta(start_t, end_t, w, num_iter), end = '\n\n')
main()