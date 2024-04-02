# index = 193141
# a1 = 5 + 1 = 6
# a2 = a3 = -1
# n-th element in vector b is sin(n * (4))
from cmath import sin
from matrix_operations import *
import matplotlib.pyplot as plt
import time

N = 941
A = generate_main_matrix(N, a1=6)
b = [sin(n * 4) for n in range(1, N + 1)]


def create_plot(residual_vector, title):
    plt.figure()
    plt.title(f"Elements of residual vector to number of iterations for {title} method")
    plt.plot(residual_vector)
    plt.yscale("log")
    plt.xlabel("number of iterations")
    plt.ylabel("residual vector element")
    plt.show()


# print_matrix(A)
# print_matrix(b)

time_start = time.time()
x_jacobi, res_jacobi, iterations_jacobi = jacobi(A, b)
time_jacobi = time.time() - time_start

time_start = time.time()
x_gauss_seidel, res_gauss_seidel, iterations_gauss_seidel = gauss_seidel(A, b)
time_gauss_seidel = time.time() - time_start

print(f"Number of iterations for Jacobi method: {iterations_jacobi}")
print(f"Computing time for Jacobi method: {round(time_jacobi, 2)}s")
create_plot(res_jacobi, "Jacobi")

print(f"Number of iterations for Jacobi method: {iterations_gauss_seidel}")
print(f"Computing time for Jacobi method: {round(time_gauss_seidel, 2)}s")
create_plot(res_gauss_seidel, "Gauss-Seidel")

