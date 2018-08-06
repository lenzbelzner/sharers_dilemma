import numpy as np
from scipy.stats import norm

seed = np.random.randint(100000)
print(seed)
np.random.seed(seed)


def f1_(x, y):
    return np.sin(x - 1) + np.cos(y)


def f2_(x, y):
    return np.cos(x + y + 1) - np.sin(x)


def f1_(x, y):
    return (1 - (x ** 2 + y ** 3)) * np.exp(-(x ** 2 + y ** 2) / 2)


d_x = np.random.uniform(-2, 2)
d_y = np.random.uniform(-2, 2)


def f1(x, y):
    return np.sin(x * y + d_x) + np.sin(x * d_y) * np.cos(y + x * d_y)


d_x2 = np.random.uniform(-2, 2)
d_y2 = np.random.uniform(-2, 2)


def f2(x, y):
    return np.sin(x * y * d_x2 * d_y2) + np.cos(x * d_x2 + y * d_y2)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def generate_f_2():
    c = np.random.uniform(-1, 1, size=4)
    o = np.random.uniform(-1, 1, size=4)
    terms = np.random.choice([np.sin, np.cos], size=2)

    print(c, o, terms)

    def f(x, y):
        return sigmoid(
            terms[0](c[0] * (x + o[0]) + c[1] * (y + o[1])) + terms[1](c[2] * (x + o[2]) + c[3] * (y + o[3])))

    return f


def generate_f(n_agents):
    n_terms = 4
    c = np.random.uniform(-1, 1, size=n_agents * n_terms)
    o = np.random.uniform(-1, 1, size=n_agents * n_terms)
    terms = np.random.choice([np.sin, np.cos], size=n_terms)

    # terms = np.random.choice([np.sin, np.cos, np.sqrt, np.exp], size=2)
    # print(c, o, terms)

    def f(x):
        return sigmoid(np.sum(
            [terms[i](
                np.sum([c[i * j] * (x[j] + o[i * j])
                        for j in range(n_agents)], axis=0))
                for i in range(n_terms)], axis=0))

    return f


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


c = np.random.uniform(1, 2, size=2)


def price(x):
    return norm(c[0], c[1]).pdf(x)


def gen_f():
    c = np.random.uniform(0, 1, size=4)
    o = np.random.uniform(0, 1, size=4)

    def cost(x):
        return np.abs(np.sum([c[i] * (x - o[i]) ** i for i in range(4)], axis=0) / (x + 0.001))

    def f(x, i):
        global_production = np.sum(x, axis=0)
        p = price(global_production)
        individual_production = x[i]
        individual_cost = cost(individual_production)
        return individual_production * sigmoid(p - individual_cost)

    return f
