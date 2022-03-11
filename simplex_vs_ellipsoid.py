import os
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, interpolate

iterations = 0


def create_klee_minty_problem(dim):
    A = []
    b = []
    objective = []
    # build the constraints matrix A
    for i in range(1, dim+1):
        constrain = [0] * dim
        for j in range(1, i):
            coefficient = 2 ** (i - j)
            if coefficient == 1:
                coefficient = 1
                constrain[j-1] = coefficient
            else:
                constrain[j-1] = coefficient * 2
        constrain[i-1] = 1
        A.append(constrain)
        b.append(4 ** (i - 1))
    # build the objective
    for j in range(dim - 1, -1, -1):
        objective.append(2**j)

    return A, b, objective


def create_random_problem(dim, min_val=0, max_val=500):
    A = []
    for _ in range(dim):
        A.append(random.sample(range(min_val, max_val), dim))
    b = random.sample(range(min_val, max_val), dim)
    objective = random.sample(range(min_val, max_val), dim)
    return A, b, objective


def callback(res):
    global iterations
    iterations += 1


def solve_klee_minty_with_simplex(dims, max_iteration):
    global iterations
    iterations = 0
    A, b, objective = create_klee_minty_problem(dims)
    return optimize.linprog(objective, method='simplex', A_ub=A, b_ub=b, callback=callback,
                            options={'maxiter': max_iteration})


def solve_random_problem_with_simplex(dims):
    global iterations
    iterations = 0
    A, b, objective = create_random_problem(dims)
    return optimize.linprog(objective, method='simplex', A_ub=A, b_ub=b, callback=callback)


def solve_klee_minty_with_interior_point(dims, max_iteration):
    global iterations
    iterations = 0
    A, b, objective = create_klee_minty_problem(dims)
    return optimize.linprog(objective, method='interior-point', A_ub=A, b_ub=b, callback=callback,
                            options={'maxiter': max_iteration})


def solve_random_problem_with_interior_point(dims):
    global iterations
    iterations = 0
    A, b, objective = create_random_problem(dims)
    return optimize.linprog(objective, method='interior-point', A_ub=A, b_ub=b, callback=callback)


def create_graph(x_data, y_data, max_dims, alg_type, problem_type):
    x = np.array(x_data)
    y = np.array(y_data)
    # make a smoother graph
    x_new = np.linspace(1, max_dims, max_dims * 100)
    a_BSpline = interpolate.make_interp_spline(x, y)
    y_new = a_BSpline(x_new)

    plt.xlabel('dimensions', fontdict={'weight': 'bold'})
    plt.ylabel('iterations', fontdict={'weight': 'bold'})
    plt.title(f'{alg_type} {problem_type}', fontdict={'weight': 'bold'})
    plt.xticks(list(range(1, max_dims + 1)))
    plt.plot(x_new, y_new, color='red')

    file_path = os.path.join('.', f'{alg_type}_{problem_type}.png')
    plt.savefig(file_path)
    plt.show()


def main():
    max_dims = 10
    max_iteration = 2 ** 10

    plot_data = []
    dim_range = list(range(1, max_dims + 1))
    for dim in dim_range:
        sum_of_iterations = 0
        res = solve_klee_minty_with_simplex(dim, max_iteration)
        sum_of_iterations += iterations
        if not res.success:
            print('klee_minty_with_simplex failed', res.message)
        plot_data.append(sum_of_iterations)
    create_graph(dim_range, plot_data, max_dims, 'simplex', 'klee-minty')

    plot_data = []
    dim_range = list(range(1, max_dims + 1))
    for dim in dim_range:
        sum_of_iterations = 0
        res = solve_random_problem_with_simplex(dim)
        sum_of_iterations += iterations
        if not res.success:
            print('random_problems_with_simplex failed', res.message)
        plot_data.append(sum_of_iterations)
    create_graph(dim_range, plot_data, max_dims, 'simplex', 'random-problems')

    plot_data = []
    dim_range = list(range(1, max_dims + 1))
    for dim in dim_range:
        sum_of_iterations = 0
        res = solve_klee_minty_with_interior_point(dim, max_iteration)
        sum_of_iterations += iterations
        if not res.success:
            print('klee_minty_with_interior_point failed', res.message)
        plot_data.append(sum_of_iterations)
    create_graph(dim_range, plot_data, max_dims, 'interior_point', 'klee-minty')
    plot_data = []
    dim_range = list(range(1, max_dims + 1))
    for dim in dim_range:
        sum_of_iterations = 0
        res = solve_random_problem_with_interior_point(dim)
        sum_of_iterations += iterations
        if not res.success:
            print('random_problems_with_interior_point failed', res.message)
        plot_data.append(sum_of_iterations)
    create_graph(dim_range, plot_data, max_dims, 'interior_point', 'random-problems')


if __name__ == '__main__':
    main()
