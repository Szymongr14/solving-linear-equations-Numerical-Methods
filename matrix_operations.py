import copy


def generate_main_matrix(n=10, a1=1, a2=-1, a3=-1):
    main_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                main_matrix[i][j] = a1
            elif i == j - 1 or i == j + 1:
                main_matrix[i][j] = a2
            elif i == j - 2 or i == j + 2:
                main_matrix[i][j] = a3
            else:
                main_matrix[i][j] = 0

    return main_matrix


def print_matrix(matrix):
    for row in matrix:
        print(row)


def max_abs_vector(v):
    return max(abs(x) for x in v)


def matrix_multiply(A, B):
    if not isinstance(A, list) or not isinstance(A[0], list):
        raise ValueError("A must be a 2D list")
    if not isinstance(B, list) or not isinstance(B[0], list):
        raise ValueError("B must be a 2D list")

    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError("Columns of A must be equal to rows of B.")

    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]

    return C


def matrix_vector_multiply(matrix, vector):
    if not isinstance(matrix, list) or not isinstance(matrix[0], list):
        raise ValueError("Matrix must be a 2D list")
    if not isinstance(vector, list):
        raise ValueError("Vector must be a 1D list")

    rows_matrix = len(matrix)
    cols_matrix = len(matrix[0])
    rows_vector = len(vector)

    if cols_matrix != rows_vector:
        raise ValueError("Number of columns in matrix must be equal to the number of rows in vector.")

    result = [0 for _ in range(rows_matrix)]

    for i in range(rows_matrix):
        for j in range(cols_matrix):
            result[i] += matrix[i][j] * vector[j]

    return result


def vector_subtract(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be the same length.")

    result = [v1[i] - v2[i] for i in range(len(v1))]
    return result


def jacobi(A, b, tol=1e-9):
    n = len(b)
    x = [0 for _ in range(n)]
    residual_errors = []
    iterations = 0

    while True:
        x_old = copy.copy(x)
        iterations += 1
        for i in range(n):
            current_sum = 0
            for j in range(n):
                if i != j:
                    current_sum += A[i][j] * x_old[j]

            x[i] = (b[i] - current_sum) / A[i][i]

        r = vector_subtract(b, matrix_vector_multiply(A, x))
        residual_errors.append(max_abs_vector(r))

        if max_abs_vector(r) < tol:
            break

    return x, residual_errors, iterations


def gauss_seidel(A, b, tol=1e-9):
    n = len(b)
    x = [0 for _ in range(n)]
    residual_errors = []
    iterations = 0

    while True:
        iterations += 1
        for i in range(n):
            current_sum = 0
            for j in range(n):
                if i != j:
                    current_sum += A[i][j] * x[j]

            x[i] = (b[i] - current_sum) / A[i][i]

        r = vector_subtract(b, matrix_vector_multiply(A, x))
        residual_errors.append(max_abs_vector(r))

        if max_abs_vector(r) < tol:
            break

    return x, residual_errors, iterations
