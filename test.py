import numpy as np

import matrix_operations

matrix1 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

matrix2 = np.array([[5, 3, 4],
                    [4, 9, 0],
                    [2, 1, 7]])

vector1 = np.array([1, 2, 3])
vector2 = np.array([1, 8, 3])

print(matrix1)
print(matrix2)

result1 = matrix1 @ matrix2
result2 = matrix1 @ vector1

print(result1)
print(result2)
print(np.dot(vector2, vector1))
print(vector2 @ vector1)

A = np.array([[1, 1, -1],
              [1, -2, 3],
              [2, 3, 1]])

b = np.array([4, -6, 7])

x = matrix_operations.solve_linear_equation_with_lu_factorization(A, b)
print(x)

r = matrix_operations.vector_subtract(b, matrix_operations.matrix_vector_multiply(A, x))


print(r)
# L, U = matrix_operations.lu_factorization(A)
# matrix_operations.print_matrix(L)
# print()
# matrix_operations.print_matrix(U)
#
# X = matrix_operations.create_identity_matrix(4)
# print(X)

