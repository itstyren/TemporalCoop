import numpy as np
from copy import deepcopy
import tools as toolbox
import configparser


# Read config
conf = configparser.ConfigParser()
conf.read('./Config_Fig5.ini', encoding='UTF-8')

# Set parameter value of strategy calculation
r = float(conf.get('param', 'r'))
c = float(conf.get('param', 'c'))
sigma = float(conf.get('param', 'sigma'))

# Set individual interaction state
# interaction rule
# 'b' is stochastic interaction, 't' is periodic interaction
switch_dis = str(conf.get('state', 'switch_dis'))
# activation probability
p = float(conf.get('state', 'p'))

# Set parameter value
agent_num = int(conf.get('network', 'agent_num'))
simulation_time = int(conf.get('network', 'simulation_time'))


def set_r_value(value):
    global r
    r = value

def set_sigma_value(value):
    global sigma
    sigma = value

def set_p_value(value):
    global p
    p = value

def set_k_value(value):
    global k
    k = value


class Agent:
    def __init__(self, index, node, graph):
        self.__index = index
        self.__node = node
        # Record the list of neighbors of node (i,j)
        self.__neighbors = list(graph.neighbors(node))
        self.__neighbors_list = []
        self.__active_neighbors_list = []
        # State: 0 is inactive, 1 is active
        self.__state_list = []
        self.__state = -1
        # Strategy: 0 is C, 1 is D
        self.__strategy = -1
        self.__payoff = 0
        self.__G = 0
        if switch_dis == 't':
            # Get the distance between the node and the center point
            self.__distance = graph.nodes[node]['distance']

    # Set the state of the individual throughout the simulation
    def set_agent_state(self):
        if switch_dis == 'b':
            self.__state_list = toolbox.gen_stochastic_distribution(simulation_time, p)
        elif switch_dis == 't':
            self.__state_list = toolbox.gen_periodic_distribution(simulation_time, p, self.__distance)

    def calculate_payoff(self, t):
        self.__payoff = 0

        # Calculate the number of active cooperator and active defector in a group
        temp_list = toolbox.count_agent_strategy(self.__active_neighbors_list)
        # plus its own strategy if active
        if self.__state == 1:
            temp_list[self.get_strategy()] += 1
        N_c, N_d = temp_list

        # Calculating gains for self-centered group
        if N_c + N_d >= 2 and N_c != 0:
            # Similar to loner when not activated, payoff = sigma
            if self.__state == 0:
                self.__payoff += sigma
            # Calculating active individual payoffs
            elif self.__strategy == 0:
                self.__payoff += c * (r * N_c - sigma * (self.__G - N_c - N_d)) / (N_c + N_d) - c
            elif self.__strategy == 1:
                self.__payoff += c * (r * N_c - sigma * (self.__G - N_c - N_d)) / (N_c + N_d)
        else:
            self.__payoff += 0

        # Sum of gains from all neighboring groups
        for neighbour_agent in self.__neighbors_list:
            temp_list = toolbox.count_agent_strategy(
                neighbour_agent.get_active_neighbors_list())
            if neighbour_agent.get_state(t) == 1:
                temp_list[neighbour_agent.get_strategy()] += 1
            N_c, N_d = temp_list

            if N_c + N_d >= 2 and N_c != 0:
                if self.__state == 0:
                    self.__payoff += sigma
                elif self.__strategy == 0:
                    self.__payoff += c * (r * N_c - sigma * (neighbour_agent.get_G() - N_c - N_d)) / (N_c + N_d) - c
                elif self.__strategy == 1:
                    self.__payoff += c * (r * N_c - sigma * (neighbour_agent.get_G() - N_c - N_d)) / (N_c + N_d)
            else:
                self.__payoff += 0

        self.__payoff = round(self.__payoff, 4)


    def set_agent_neighbors_list(self, agent_list):
        for index, node in enumerate(self.__neighbors):
            self.__neighbors_list.append(
                [agent for agent in agent_list if agent.get_node() == node][0])
        self.__G = len(self.__neighbors) + 1

    def set_active_neighbors_list(self, t):
        self.__active_neighbors_list = []
        for neighbor in self.__neighbors_list:
            if neighbor.get_state(t) == 1:
                self.__active_neighbors_list.append([neighbor][0])

    def set_init_strategy(self, strategy):
        self.__strategy = strategy

    # Get the state of the individual at time t
    def get_state(self, t):
        self.__state = self.__state_list[t]
        return self.__state

    def get_strategy(self):
        return self.__strategy

    def get_payoff(self):
        return self.__payoff

    def get_neighbors_list(self):
        return self.__neighbors_list

    def get_active_neighbors_list(self):
        return self.__active_neighbors_list

    def get_node(self):
        return self.__node

    def get_G(self):
        return self.__G

    def get_index(self):
        return self.__index

    # Randomly selects a neighbor update strategy
    def update_strategy(self, t):
        N_c, N_d = toolbox.count_agent_strategy(self.__neighbors_list)
        choice_list = []

        for index, neighbour in enumerate(self.__neighbors_list):
            choice_list.append(index)

        if len(choice_list) != 0:
            neighbour_agent = self.__neighbors_list[np.random.choice(choice_list)]

            neighbour_agent.calculate_payoff(t)

            p_i_to_j = toolbox.get_p_i_to_j(self.__payoff, neighbour_agent.get_payoff())

            res = np.random.rand()
            if res < p_i_to_j:
                self.__strategy = neighbour_agent.get_strategy()


# Record the change in each time
class Recorder:
    def __init__(self):
        self.__agent_strategy_list = np.zeros((simulation_time, agent_num)).astype("int8")
        self.__agent_state_list = np.zeros((simulation_time, agent_num)).astype("int8")
        self.__strategy_fraction_list = []
        self.__state_fraction_list = []
        self.__agent_num = agent_num
        self.__total_time = 0
        self.__agent_list = []


    def recording(self, agent_list, t):
        agent_strategy_this_time = []
        agent_state_this_time = []

        for i in range(self.__agent_num):
            agent_strategy_this_time.append(agent_list[i].get_strategy())
            agent_state_this_time.append(agent_list[i].get_state(t))

        list_strategy_fraction_this_time = toolbox.count_agent_strategy_fraction(agent_list)
        list_state_fraction_this_time = toolbox.count_agent_state_fraction(agent_list, t)

        self.__record_agent_strategy(agent_strategy_this_time, t)
        self.__record_agent_state(agent_state_this_time, t)
        self.__record_list_strategy_fractions(list_strategy_fraction_this_time)
        self.__record_list_state_fractions(list_state_fraction_this_time)
        self.__total_time += 1

    def __record_agent_strategy(self, agent_strategy_this_time, t):
        self.__agent_strategy_list[t] = agent_strategy_this_time

    def __record_agent_state(self, agent_state_this_time, t):
        self.__agent_state_list[t] = agent_state_this_time

    def __record_list_strategy_fractions(self, list_strategy_fraction_this_time):
        self.__strategy_fraction_list.append(
            list_strategy_fraction_this_time)

    def __record_list_state_fractions(self, list_state_fraction_this_time):
        self.__state_fraction_list.append(
            list_state_fraction_this_time)

    def __record_agent_list(self, agent_list):
        self.__agent_list.append(deepcopy(agent_list))

    def get_total_list_strategy_fraction(self):
        return self.__strategy_fraction_list

    def get_total_list_state_fraction(self):
        return self.__state_fraction_list

    def get_agent_strategy_list(self):
        return self.__agent_strategy_list

    def get_agent_state_list(self):
        return self.__agent_state_list

    def get_agent_list(self):
        return self.__agent_list