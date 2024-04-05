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


def create_plot(first_residual_vector, title, second_residual_vector=None, first_label=None, second_label=None,
                lower_bound=1e-9, upper_bound=1e10):
    plt.figure()
    # plt.title(f"Elements of residual vector to number of iterations for {title} method")
    plt.plot(first_residual_vector, label=first_label)
    if second_residual_vector is not None:
        plt.plot(second_residual_vector, color="orange", label=second_label)
    # plt.axhline(y=lower_bound, color='g', linestyle='dashed', label="lower boundary")
    plt.axhline(y=upper_bound, color='r', linestyle='dashed', label="upper boundary")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("number of iterations")
    plt.ylabel("residual vector element")
    plt.show()


# print_matrix(A)
# print_matrix(b)

# task B
time_start = time.time()
x_jacobi, res_jacobi, iterations_jacobi = jacobi(A, b)
time_jacobi = time.time() - time_start

time_start = time.time()
x_gauss_seidel, res_gauss_seidel, iterations_gauss_seidel = gauss_seidel(A, b)
time_gauss_seidel = time.time() - time_start

print(f"Number of iterations for Jacobi method: {iterations_jacobi}")
print(f"Computing time for Jacobi method: {round(time_jacobi, 2)}s")
create_plot(res_jacobi, "Jacobi", None, "Jacobi method")

print(f"Number of iterations for Gauss-Seidel method: {iterations_gauss_seidel}")
print(f"Computing time for Gauss-Seidel method: {round(time_gauss_seidel, 2)}s")
create_plot(res_gauss_seidel, "Gauss-Seidel", None, "Gauss-Seidel method")

create_plot(res_gauss_seidel, "comparison", res_jacobi, "gauss-seidel method", "jacobi method")


# # task C
A = generate_main_matrix(N, a1=3)
x_jacobi1, res_jacobi1, iterations_jacobi1 = jacobi(A, b)
create_plot(res_jacobi1, "Jacobi", None, "Jacobi method")
print(f"Jacobi iter: {iterations_jacobi1}")

x_gauss_seidel1, res_gauss_seidel1, iterations_gauss_seidel1 = gauss_seidel(A, b)
create_plot(res_gauss_seidel1, "Gauss-Seidel", None, "Gauss-Seidel method")
print(f"Gauss iter: {iterations_gauss_seidel1}")

create_plot(res_gauss_seidel1, "comparison", res_jacobi1, "gauss-seidel method", "jacobi method")

# task D
A = generate_main_matrix(N, a1=3)
x_lu = solve_linear_equation_with_lu_factorization(A, b)
print(x_lu)
res = vector_subtract(b, matrix_vector_multiply(A, x_lu))
norm = max_abs_vector(res)
print(res)
print(norm)

# task E
i = 100
jacobi_times = []
gauss_seidel_times = []
lu_times = []
sizes = []
while i <= 3001:
    A = generate_main_matrix(i, a1=6)
    b = [sin(n * 4) for n in range(1, i + 1)]
    sizes.append(i)

    time_start = time.time()
    x_jacobi, res_jacobi, iterations_jacobi = jacobi(A, b)
    end = time.time() - time_start
    jacobi_times.append(end)

    time_start = time.time()
    x_gauss_seidel, res_gauss_seidel, iterations_gauss_seidel = gauss_seidel(A, b)
    end = time.time() - time_start
    gauss_seidel_times.append(end)

    time_start = time.time()
    x_lu = solve_linear_equation_with_lu_factorization(A, b)
    end = time.time() - time_start
    lu_times.append(end)

    i *= 2

plt.figure()
plt.plot(sizes, jacobi_times, label="Jacobi method")
plt.plot(sizes, gauss_seidel_times, label="Gauss-Seidel method")
plt.plot(sizes, lu_times, label="LU factorization")
plt.legend()
plt.xlabel("matrix size")
plt.ylabel("computing time [s]")
plt.show()


