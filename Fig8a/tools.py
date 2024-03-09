import math
import numpy as np
import networkx as nx
import configparser, inspect

conf = configparser.ConfigParser()
conf.read('./Config_Fig8a.ini', encoding='UTF-8')
k = float(conf.get('param', 'k'))
p = float(conf.get('state', 'p'))
agent_num = int(conf.get('network', 'agent_num'))
# interaction rule
switch_dis = str(conf.get('state', 'switch_dis'))


# Find items whose variable names/values match the input var
def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [
            var_name for var_name, var_val in fi.frame.f_locals.items()
            if var_val is var
        ]
        if len(names) > 0:
            return names[0]


# Calculate the number of C D in agent list
def count_agent_strategy(agent_list):
    agent_strategy_num_list = [0, 0]
    for i in range(len(agent_list)):
        agent_strategy_num_list[agent_list[i].get_strategy()] += 1
    return agent_strategy_num_list  # [C count, D count]


# Calculate the strategy fraction
def count_agent_strategy_fraction(agent_list):
    agent_strategy_fraction_list = np.array([0, 0])
    for i in range(len(agent_list)):
        agent_strategy_fraction_list[agent_list[i].get_strategy()] += 1
    return agent_strategy_fraction_list / np.sum(agent_strategy_fraction_list)

def count_agent_state_fraction(agent_list, t):
    agent_state_fraction_list = np.array([0, 0])
    for i in range(len(agent_list)):
        agent_state_fraction_list[agent_list[i].get_state(t)] += 1
    return agent_state_fraction_list / np.sum(agent_state_fraction_list)

# Fermi function
def get_p_i_to_j(u_i, u_j):
    u_i = round(u_i, 4)
    u_j = round(u_j, 4)
    try:
        res = 1 / (1 + math.exp(u_i - u_j) / k)
    except OverflowError:
        res = 1
    return res


# Generate initial network
def gen_regular_network(n):
    # square lattice
    net = nx.grid_2d_graph(int(pow(n, 0.5)), int(pow(n, 0.5)), periodic=True)
    return net

def init_regular_network(agent_num):
    graph = gen_regular_network(agent_num)
    # for temporal interaction, calculate distance between nodes
    if switch_dis == 't':
        # set a center point to represent the simulation time
        x0 = y0 = np.sqrt(agent_num) / 2
        for node in graph.nodes:
            x_coordinate, y_coordinate = node
            graph.nodes[node]['x'] = x_coordinate
            graph.nodes[node]['y'] = y_coordinate
            euclidean_distance = np.sqrt(
                np.square(x0 - graph.nodes[node]['x'])
                + np.square(y0 - graph.nodes[node]['y']))
            # Adjust the Euclidean distance according to scale
            # The distance between the farthest node and the center is about 12
            scale = np.around(12 / (1.414 * x0), decimals=2)
            distance = euclidean_distance * scale
            graph.nodes[node]['distance'] = np.around(distance, decimals=2)
    return graph


def gen_regular_network_vary_degree(k, n):
    net = nx.random_regular_graph(k, n)
    return net

def init_regular_network_var_degree(degree, agent_num):
    graph = gen_regular_network_vary_degree(degree, agent_num)
    # Generate node locations
    pos = nx.spring_layout(graph)
    # Add the horizontal and vertical coordinates to the node properties
    for node, coordinates in pos.items():
        x_coordinate = coordinates[0]
        y_coordinate = coordinates[1]
        # x, y belong to [-1,1], with [0,0] as the center of the graph
        euclidean_distance = np.around(np.sqrt(x_coordinate ** 2 + y_coordinate ** 2), decimals=4)
        # Adjust the Euclidean distance according to scale
        scale = np.around(12 / 1.414, decimals=2)
        distance = euclidean_distance * scale
        graph.nodes[node]['distance'] = np.around(distance, decimals=2)
    return graph


def gen_ER_network(n):
    # Erdos-Renyi graph
    net = nx.erdos_renyi_graph(n, p=4/n)
    return net

def init_ER_network(agent_num):
    graph = gen_ER_network(agent_num)
    pos = nx.spring_layout(graph)
    for node, coordinates in pos.items():
        x_coordinate = coordinates[0]
        y_coordinate = coordinates[1]
        euclidean_distance = np.around(np.sqrt(x_coordinate ** 2 + y_coordinate ** 2), decimals=4)
        scale = np.around(12 / 1.414, decimals=2)
        distance = euclidean_distance * scale
        graph.nodes[node]['distance'] = np.around(distance, decimals=2)
    return graph


def gen_sw_network(n, k=4, p=0.1):
    # small world graph
    net = nx.watts_strogatz_graph(n, k, p)
    return net

def init_sw_network(agent_num):
    graph = gen_sw_network(agent_num, 4, 0.5)
    pos = nx.spring_layout(graph)
    for node, coordinates in pos.items():
        x_coordinate = coordinates[0]
        y_coordinate = coordinates[1]
        euclidean_distance = np.around(np.sqrt(x_coordinate ** 2 + y_coordinate ** 2), decimals=4)
        scale = np.around(12 / 1.414, decimals=2)
        distance = euclidean_distance * scale
        graph.nodes[node]['distance'] = np.around(distance, decimals=2)
    return graph


def gen_ba_network(n, k=4):
    # Barabasi-Albert graph
    net = nx.random_graphs.barabasi_albert_graph(n, k)
    return net

def init_ba_network(agent_num):
    graph = gen_ba_network(agent_num, 4)
    pos = nx.spring_layout(graph)
    for node, coordinates in pos.items():
        x_coordinate = coordinates[0]
        y_coordinate = coordinates[1]
        euclidean_distance = np.around(np.sqrt(x_coordinate ** 2 + y_coordinate ** 2), decimals=4)
        scale = np.around(12 / 1.414, decimals=2)
        distance = euclidean_distance * scale
        graph.nodes[node]['distance'] = np.around(distance, decimals=2)
    return graph


def gen_network_set():
    if switch_dis == 'b':
        network_set = [init_regular_network(agent_num), init_ER_network(agent_num),
                       init_sw_network(agent_num), init_ba_network(agent_num)]
    else:
        network_set = [init_regular_network_var_degree(3, agent_num),
                       init_regular_network(agent_num),
                       init_regular_network_var_degree(6, agent_num),
                       init_regular_network_var_degree(8, agent_num)]
    return network_set


# Generate state list of an individual throughout the simulation
# stochastic interaction obey the Bernoulli distribution
def gen_stochastic_distribution(simulation_time, p):
    state = [1]
    state.extend((np.random.binomial(n=1, p=p, size=simulation_time - 1)).tolist())
    return state


# periodic interaction
def gen_periodic_distribution(simulation_time, p, distance):
    # Let 24 be a period
    T = 24

    # activation_time is Poisson distribution
    # lam is the average number of events per unit of time, simulation_time = lam*size
    # p is average activation probability, p*T is average length of activation in a period
    active_time = np.random.poisson(lam=p*T, size=int(simulation_time / T) + 1)

    # The time difference is normally distributed
    # the mean is the distance of the node from the center of the graph, and the variance is 2
    state = np.zeros(simulation_time)
    jat_lag = np.around(np.random.normal(loc=distance, scale=2, size=1))

    # generate individual state list
    index = 0
    state[0] = 1
    for i in range(len(active_time)):
        wake = int(index + jat_lag)
        if wake < simulation_time:
            sleep = int(wake + active_time[i])
            state[wake:sleep] = 1
        index += T
    return state.astype("uint32").tolist()