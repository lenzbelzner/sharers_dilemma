import numpy as np


def gen_logistic():
    # c = np.random.uniform(0, 2, size=2)
    c = np.random.uniform(1, 3)
    o = np.random.uniform(1, 3)

    def logistic(x):
        # return 1 / (1 + np.exp(-(c[0] + c[1] * (x - o))))
        return 1 / (1 + np.exp(- c * (x - o)))

    return logistic


def gen_inv_logistic():
    log = gen_logistic()

    def inv_logistic(x):
        return 1 - log(x)

    return inv_logistic


class Agent:
    def __init__(self):
        self.prod_a = gen_inv_logistic()
        self.prod_b = gen_logistic()


class Market:
    def __init__(self, n_agents):
        self.price_a = gen_inv_logistic()
        self.price_b = gen_inv_logistic()
        self.agents = [Agent() for _ in range(n_agents)]

    def utilities(self, joint_x):
        # global_a = np.sum([agent.prod_a(joint_x[i]) for i, agent in enumerate(self.agents)], axis=0)
        global_b = np.sum([agent.prod_b(joint_x[i]) for i, agent in enumerate(self.agents)], axis=0)
        utilities = {}
        for i, agent in enumerate(self.agents):
            agent_x = joint_x[i]
            # u = agent.prod_a(agent_x) * self.price_a(global_a) + agent.prod_b(agent_x) * self.price_b(global_b)
            u = agent.prod_b(agent_x) * self.price_b(global_b)
            utilities[i] = u
        return utilities
