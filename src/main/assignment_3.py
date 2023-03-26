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

    for i in range(n):
        for j in range(i+1, n):
            A[j] = A[j] - (A[j][i] / A[i][i]) * A[i]

    return A

def BackSubstitution(A):
    n = len(A)
    x = np.zeros(n)

    for i in range(n):
        sum = 0
        for j in range(n-i-1, n):
            sum += A[n-i-1][j] * x[j]
        x[n-i-1] = (A[n-i-1][n] - sum) / A[n-i-1][n-i-1]

    return x.astype(int)
# End of Question 3

# Question 4
#
def Determinant(A, det):
    for i in range(len(A)):
        if (len(A) != 2):
            B = np.delete(A, 0, 0)
            B = np.delete(B, i, 1)
            det += (-1) ** i * A[0][i] * Determinant(B, det)
        else:
            return A[0][0] * A[1][1] - A[0][1] * A[1][0]

    return det

def L_fact(A, U):
    n = len(A)
    m = len(A[0])
    L = np.zeros((n, m))
    
    for i in range(n):
        L[i][i] = 1
    for i in range(1, n):
        for j in range(i):
            diff = A[i][j]
            for k in range(j):
                diff -= L[i][k] * U[k][j]
            L[i][j] = diff / U[j][j]

    return L
# End of Question 4

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
    A_b = np.array([[2., -1, 1, 6],
                    [1, 3, 1, 0],
                    [-1, 5, 4, -3]])

    # Printing result of Gaussian Elimination and back substitution
    A_b_elim = GaussianElim(A_b)
    print(BackSubstitution(A_b_elim), end = '\n\n')

    # Initializing matrix for LU Factorization
    matrix_LU = np.array([[1., 1, 0, 3],
                          [2, 1, -1, 1],
                          [3, -1, -1, 2],
                          [-1, 2, 3, -1]])
    U = matrix_LU.copy()
    
    print(Determinant(matrix_LU, 0), end = '\n\n')
    
    GaussianElim(U)
    print(L_fact(matrix_LU, U), end = '\n\n')
    
    print(U, end = '\n\n')
    

main()