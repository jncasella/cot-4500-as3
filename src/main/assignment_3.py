import numpy as np

# Function for both Euler Method and Runge-Kutta method
def function(t, w):
    return t - (w ** 2)

# Question 1
#
def Euler(t, end_t, w, num_iter):
    h = (end_t - t) / num_iter
    
    for i in range(num_iter):        
        w += h * function(t, w)
        t += h
    
    return w
# End of Question 1

# Question 2
#
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
# End of Question 2

# Question 3
#
def GaussianElim(A):
    n = len(A)
    x = np.zeros(n)
    
    for i in range(n):
        for j in range(i+1, n):
            A[j] = A[j] - (A[j][i] / A[i][i]) * A[i]
    
    for i in range(n):
        sum = 0
        for j in range(n-i-1, n):
            sum += A[n-i-1][j] * x[j]
        x[n-i-1] = (A[n-i-1][n] - sum) / A[n-i-1][n-i-1]
    
    return x.astype(int)
# End of Question 3

def main():
    # Starting information for Question 1 and 2
    start_t = 0
    end_t = 2
    w = 1
    num_iter = 10
    
    # Printing result of Euler method
    print(Euler(start_t, end_t, w, num_iter), end = '\n\n')
    
    # Printing result of Runge-Kutta method
    print(RungeKutta(start_t, end_t, w, num_iter), end = '\n\n')
    
    # Initializing matrix for Gaussian Elimination
    A_b = np.array([[2.0, -1, 1, 6],
                    [1, 3, 1, 0],
                    [-1, 5, 4, -3]])
    
    # Printing result of Gaussian Elimination and back substitution
    print(GaussianElim(A_b))
    
main()