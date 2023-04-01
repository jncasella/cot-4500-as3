import numpy as np

# Function for both Euler Method and Runge-Kutta method
def function(t, w):
    return t - (w ** 2)

# Question 1
# Euler Function
# Takes in four parameters:
#       * t is the lower bound of the interval in which the
#               Euler method will be applied
#       * end_t is the upper bound of the same interval
#       * w is the starting value of the approximation
#       * num_iter is the number of iterations for which the
#               Euler method is to be carried out
# Finds the approximate solution to a differential equation
#       given in the function function defined above.
# Returns the value of the approximate function at the point end_t
def Euler(t, end_t, w, num_iter):
    h = (end_t - t) / num_iter

    for i in range(num_iter):
        w += h * function(t, w)
        t += h

    return w
# End of Question 1

# Question 2
# RungeKutta Function
# Takes in four parameters:
#       * t is the lower bound of the interval in which the
#               Runge-Kutta method will be applied
#       * end_t is the upper bound of the same interval
#       * w is the starting value of the approximation
#       * num_iter is the number of iterations for which the
#               Runge-Kutta method is to be carried out
# Estimates the solution to the differential equation as defined
#       in the function function by applying the fourth order
#       Runge-Kutta method
# Returns the value of the approximate function at the point end_t
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
# GaussianElim Function
# Takes in a two-dimensional NumPy array
#       It is assumed that the array represents a consistent system
# Performs Gaussian elimination to reduce the matrix into an upper
#       triangular matrix
# Returns the reduced matrix
def GaussianElim(A):
    n = len(A)

    for i in range(n):
        for j in range(i+1, n):
            A[j] = A[j] - (A[j][i] / A[i][i]) * A[i]

    return A

# BackSubstitution Function
# Takes in a two dimensional array
#       It is assumed that this array has already been put through
#           the GaussianElim function
#       It is also assumed that the matrix passed represents a
#           consistent system
# Performs backward substitution to find the solutions to the system
#       represented by the matrix
# Returns an one-dimensional array of the solutions to the system
def BackSubstitution(A):
    n = len(A)
    x = np.zeros(n)

    for i in range(n):
        sum = 0
        for j in range(n-i-1, n):
            sum += A[n-i-1][j] * x[j]
        x[n-i-1] = (A[n-i-1][n] - sum) / A[n-i-1][n-i-1]

    return x
# End of Question 3

# Question 4
# Determinant Function
# Takes in two parameters:
#       * A two dimensional array
#               This array is assumed to be sqaure
#       * A float value
#               It is expected for this value to be zero upon the
#                   the initial call of the function
# Performs a recursive calculation of the determinant of the matrix A
# Returns the determinant of the matrix as a float
def Determinant(A, det):
    for i in range(len(A)):
        if (len(A) != 2):
            B = np.delete(A, 0, 0)
            B = np.delete(B, i, 1)
            det += (-1) ** i * A[0][i] * Determinant(B, det)
        else:
            return A[0][0] * A[1][1] - A[0][1] * A[1][0]

    return det

# L_fact Function
# Takes in two parameters
#       * A matrix A
#       * A matrix U which is assumed to the matrix A passed
#           through the GaussianElim function
# Finds the L matrix for the LU factorization of the matrix A
# Returns the matrix L
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

# Question 5
# isDiagDom Function
# Takes in a two-dimensional NumPy array, A
#           It is assumed that the matrix A is square
# Determines if the matrix A is diagonally dominant
# Returns a boolean value representing if the matrix A
#       is diagonally dominant
def isDiagDom(A):
    n = len(A)
    
    for i in range(n):
        sum = -abs(A[i][i])
        for j in range(n):
            sum += abs(A[i][j])
        if (sum > abs(A[i][i])):
            return False
    return True
# End of Question 5

# Question 6
# isPosDef Function
# Takes in a matrix A which is assumed to be sqaure
# Determines if the matrix A is a positive definite by using the
#       fact that a matrix A is positive definite if and only if
#       the matrix is symmetric and has all positive real eigenvalues
# Returns a boolean value repsenting if the matrix A is
#       positive dominant
def isPosDef(A):
    if (np.array_equal(np.transpose(A), A) == False):
        return False
    eigval, eigvec = np.linalg.eig(A)
    for i in eigval:
        if (i <= 0):
            return False
    return True
# End of Question 6

# Main Function
def main():
    # Starting information for Question 1 and 2
    start_t = 0
    end_t = 2
    w = 1
    num_iter = 10

    # Printing result of Euler method
    print("%.5f" % Euler(start_t, end_t, w, num_iter), end = '\n\n')

    # Printing result of Runge-Kutta method
    print("%.5f" % RungeKutta(start_t, end_t, w, num_iter), end = '\n\n')

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
    
    # Prints the determinant of the matrix_LU matrix
    print("%.5f" % Determinant(matrix_LU, 0), end = '\n\n')
    
    # Performs Gaussian elimination on the matrix_LU matrix to find
    #       the U matrix for LU factorization
    GaussianElim(U)
    
    # Prints the L matrix for LU factorization
    print(L_fact(matrix_LU, U), end = '\n\n')
    
    # Prints the U matrix for LU factorization
    print(U, end = '\n\n')
    
    # Initializing matrix for diagonally dominant testing
    matrix_diagDom = np.array([[9, 0, 5, 2, 1],
                               [3, 9, 1, 2, 1],
                               [0, 1, 7, 2, 3],
                               [4, 2, 3, 12, 2],
                               [3, 2, 4, 0, 8]])
    
    # Printing if the matrix_diagDom matrix is diagonally dominant
    print(isDiagDom(matrix_diagDom), end = '\n\n')
    
    # Initializing matrix to test if it is positive definite
    matrix_posDef = np.array([[2, 2, 1],
                              [2, 3, 0],
                              [1, 0, 2]])
    
    # Printing if the matrix_posDef matrix is positive definite
    print(isPosDef(matrix_posDef))

# Calling the main function
main()