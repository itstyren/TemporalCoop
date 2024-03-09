import configparser, pickle, datetime, os, shutil, random, time
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from scipy.ndimage import gaussian_filter
from functools import partial
import multiprocessing as mp
from class_record import *
import tools as toolbox

# Read config
conf = configparser.ConfigParser()
conf.read('./Config_Fig8a.ini', encoding='UTF-8')

# Set parameter value
agent_num = int(conf.get('network', 'agent_num'))
simulation_time = int(conf.get('network', 'simulation_time'))
initial_ratio = eval(conf.get('param', 'initial_ratio'))
num_processes = int(conf.get('pool', 'num_processes'))

# Set plot function
color_set = eval(conf.get('plot', 'color_set'))
# 'hpc' for calculation
# 'hpc_to_local' for plotting
mode = str(conf.get('func', 'mode'))
switch_fuc = str(conf.get('func', 'switch_fuc'))

# iterator value
r_iterator_list = eval(conf.get('iterator', 'r_iterator_list'))
sigma_iterator_list = eval(conf.get('iterator', 'sigma_iterator_list'))
p_iterator_list = eval(conf.get('iterator', 'p_iterator_list'))
k_iterator_list = eval(conf.get('iterator', 'k_iterator_list'))

# set save path
res_dir = "../result"
plot_dir = "../plot"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)


def run_simulation_show_destination_compris_network_with_two_param(limit_iterator_list,
                                                                   unlimit_lterator_list,
                                                                   param_name_1, param_name_2):
    pool = mp.Pool(num_processes)
    different_graph = toolbox.gen_network_set()

    # generate agent list on different graphs
    agent_list_different_graph = []

    for graph in different_graph:
        agent_list = []
        for i, node in enumerate(graph):
            agent_list.append(Agent(i, node, graph))
        set_init_agent_strategy(agent_list)

        agent_list_different_graph.append(agent_list)

    strategy_destination_all_graph = []
    for limit_iterator_index, limit_iterator_value in enumerate(
            limit_iterator_list):
        strategy_destination_each_sub_graph = []

        for graph_index, agent_list in enumerate(
                agent_list_different_graph):
            param_dict = {
                param_name_1: limit_iterator_value,
            }

            partial_func = partial(
                parallel_all_process_and_destination, param_dict,
                param_name_2, agent_list)

            parallel_list = pool.map(partial_func, unlimit_lterator_list)

            single_strategy_destination = []
            for single_parallel in parallel_list:
                single_strategy_destination.append(single_parallel[0])

            strategy_destination_each_sub_graph.append(
                single_strategy_destination)

        strategy_destination_all_graph.append(
            strategy_destination_each_sub_graph)
    pool.close()
    return strategy_destination_all_graph


# plot Fig_7a or Fig_8a
def plot_param_as_function_compare_three_param(limit_iterator_list_1,
                                               limit_iterator_list_2,
                                               unlimit_iterator_list,
                                               list_name_1, list_name_2,
                                               list_name_3):
    run_time = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    plot_save_dir = os.path.join(plot_dir, run_time)
    if mode != 'hpc_to_local':
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)

    config_src = 'Config_Fig8a.ini'  # or 'Config_Fig8a.ini'
    config_dst = os.path.join(plot_save_dir, run_time + "-" + config_src)

    save_name = "param_as_function_with_three_param"
    save_path = os.path.join(res_dir, save_name)

    strategy_destination = []
    if mode == 'hpc':
        strategy_destination = run_simulation_show_destination_compris_network_with_two_param(
            limit_iterator_list_1, unlimit_iterator_list, list_name_1.split('_')[0],
            list_name_3.split('_')[0])

        # Save simulation result
        with open(save_path, 'wb') as f:
            pickle.dump(strategy_destination, f)
        # Save Config file
        shutil.copyfile(config_src, config_dst)

        # record the config file path to get plot
        with open("last_save_dir.txt", "w") as f:
            f.write(plot_save_dir + "\n")

    # Read simulation result
    if mode == 'hpc_to_local':
        with open(save_path, 'rb') as f:
            strategy_destination = pickle.load(f)

        Nr = 1
        Nc = len(limit_iterator_list_1)
        fig_1 = plt.figure(num='fig_1', figsize=(9, 6))
        fig_1.subplots_adjust(
            top=0.85, bottom=0.25, left=0.15, right=0.85, hspace=0.2, wspace=0.2
        )
        fig_1_axes = fig_1.subplots(Nr, Nc, squeeze=False)

        # Get plot data
        graph_list_with_cooperation_fraction = []
        for graph in strategy_destination:
            all_curve_list = []
            for cruve_value in graph:
                x_list = []
                for x_value in cruve_value:
                    x_list.append(
                        np.around(x_value[0], decimals=2))  # only record the proportion of C
                all_curve_list.append(x_list)
            graph_list_with_cooperation_fraction.append(all_curve_list)

        labels = ['k=3', 'k=4', 'k=6', 'k=8']

        list_index = 0
        for i in range(Nr):
            for j in range(Nc):
                fig_1_axes[i][j].set_ylim(-0.05, 1.05)
                fig_1_axes[i, j].set_xlim(unlimit_iterator_list[0] - 0.05,
                                          unlimit_iterator_list[-1] + 0.05)
                for cruve_index in range(len(limit_iterator_list_2)):
                    fig_1_axes[i, j].plot(
                        unlimit_iterator_list,
                        gaussian_filter(
                            graph_list_with_cooperation_fraction[list_index]
                            [cruve_index],
                            sigma=0.5),
                        color=color_set[cruve_index],
                        marker='.',
                    )
                fig_1_axes[i][j].set_title('%s = %s' %
                                           (list_name_1.split('_')[0],
                                            limit_iterator_list_1[list_index]))
                fig_1_axes[i][j].set_xlabel(r'$%s$' %
                                            (list_name_3.split('_')[0]),
                                            fontsize=16)

                if Nc == 2:
                    fig_1.legend(labels=labels,
                                 handlelength=6,
                                 handleheight=0.2,
                                 loc='center',
                                 ncol=4,
                                 fontsize=12.5,
                                 columnspacing=4,
                                 frameon=False,
                                 bbox_to_anchor=(0.51, 0.95, 0, 0))
                else:
                    fig_1_axes[i, j].legend(labels=labels, loc='upper left')
                fig_1_axes[i][j].set_ylabel(r'$\rho_{C}$', fontsize=16)

                fig_1_axes[i][j].set_xticks(
                    np.linspace(
                        unlimit_iterator_list[0], unlimit_iterator_list[-1], 5))
                fig_1_axes[i][j].set_xticklabels(np.around(np.linspace(
                    unlimit_iterator_list[0], unlimit_iterator_list[-1], 5),
                    decimals=1),
                    fontsize=16)
                fig_1_axes[i][j].set_yticks(np.linspace(0, 1, 5).round(decimals=2))
                fig_1_axes[i][j].set_yticklabels(np.linspace(0, 1, 5).round(decimals=2),
                                                 fontsize=16)
                list_index += 1

        with open("last_save_dir.txt", "r") as f:
            last_runtime = f.readline()[:-1]

        fig_d_name = os.path.join(last_runtime, 'd-' + run_time + '.svg')
        fig_1.savefig(fig_d_name)
        plt.show(block=True)
        print('plot save')


# Simulation process of evolutionary dynamics
def set_init_agent_strategy(agent_list):
    for i in range(len(agent_list)):
        # Cooperator is 0, Defector is 1
        strategy = np.random.choice([0, 1], p=initial_ratio.ravel())
        agent_list[i].set_init_strategy(strategy)


def parallel_all_process_and_destination(param_dict, param_name, agent_list,
                                         iterator_value):
    if param_dict:
        for name in param_dict:
            exec_str = 'set_param_value(param_dict[name])'
            exec(exec_str.replace('param', name, 1))

    if type(param_name) == tuple:
        for i, name in enumerate(param_name):
            exec_str = 'set_param_value(iterator_value[i])'
            exec(exec_str.replace('param', param_name[i]))
    else:
        exec_str = 'set_param_value(iterator_value)'
        exec(exec_str.replace('param', param_name))

    t1 = time.time()

    agent_list_copy = deepcopy(agent_list)

    for i in range(agent_num):
        # Set state list
        agent_list_copy[i].set_agent_state()
        # Add neighbor list
        agent_list_copy[i].set_agent_neighbors_list(agent_list_copy)

    # Record the strategy ratio at time t and the strategies of all nodes
    record_temp = Recorder()
    # List of recorded policy ratios
    list_strategy = []

    # Monte Carlo simulation
    for t in range(simulation_time):
        for index in range(agent_num):
            # Get the state of the individual at time t
            agent_list_copy[index].get_state(t)
            # Setting the list of active neighbors at time t
            agent_list_copy[index].set_active_neighbors_list(t)

            list_strategy_this_time = toolbox.count_agent_strategy_fraction(
                agent_list_copy)
            list_strategy.append(list_strategy_this_time)
            # If all individuals are the same strategy or strategy D is 0, break
            if 1 in list_strategy_this_time or list_strategy_this_time[1] == 0:
                break

        # disorder
        arr_order = np.arange(agent_num)
        np.random.shuffle(arr_order)

        # Calculate payoff in order of shuffled individual numbers
        for i in arr_order:
            random_agent = agent_list_copy[i]
            random_agent.calculate_payoff(t)

        # Activated individuals update their strategies
        # inactive individuals keep their previous round of strategies
        for i in arr_order:
            random_agent = agent_list_copy[i]
            if random_agent.get_state(t) == 1:
                random_agent.update_strategy(t)

    t2 = time.time()

    total_time = np.rint(t2 - t1)

    print(param_name, iterator_value, "cost time：", total_time,
          's\t time', time.strftime('%Y-%m-%d %H:%M:%S'))

    # Return simulation results
    if type(param_name) == tuple:
        return [
            toolbox.count_agent_strategy_fraction(agent_list_copy),
            iterator_value
        ]
    return [
        toolbox.count_agent_strategy_fraction(agent_list_copy),
        record_temp
    ]


def main():
    if switch_fuc == 'g':
        plot_param_as_function_compare_three_param(
            sigma_iterator_list, p_iterator_list, r_iterator_list,
            toolbox.retrieve_name(sigma_iterator_list),
            toolbox.retrieve_name(p_iterator_list),
            toolbox.retrieve_name(r_iterator_list))


if __name__ == "__main__":
    main()
