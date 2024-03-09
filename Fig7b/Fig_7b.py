import configparser, pickle, datetime, os, shutil, time
import matplotlib.pyplot as plt
from matplotlib import colors
from functools import partial
import multiprocessing as mp
from class_record import *
import tools as toolbox

# Read config
conf = configparser.ConfigParser()
conf.read('./Config_Fig7b.ini', encoding='UTF-8')

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

# Plot type
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


def parallel_all_process_and_destination(param_dict, param_name, agent_list,
                                         iterator_value):

    if param_dict:
        for name in param_dict:
            exec_str = 'set_param_value(param_dict[name])'
            exec(exec_str.replace('param', name, 1))

    # set agent_list as global variable
    agent_list = iterator_value

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

        # record all agents' state
        record_temp.recording(agent_list_copy, t)

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


# Simulation process of evolutionary dynamics
def set_init_agent_strategy(agent_list):
    for i in range(len(agent_list)):
        # Cooperator is 0, Defector is 1
        strategy = np.random.choice([0, 1], p=initial_ratio.ravel())
        agent_list[i].set_init_strategy(strategy)


def run_simulation_compris_network_show_all_process():
    pool = mp.Pool(num_processes)
    # generate iteration network
    different_graph = toolbox.gen_network_set()

    agent_list_different_graph = []
    all_record_list = []

    # Set iterable agent list according to generated graph
    for graph in different_graph:
        agent_list = []
        for i, node in enumerate(graph):
            agent_list.append(Agent(i, node, graph))
        set_init_agent_strategy(agent_list)

        agent_list_different_graph.append(agent_list)

    param_dict = {}
    partial_func = partial(parallel_all_process_and_destination,
                           param_dict, 'network', [])

    parallel_list = pool.map(partial_func, agent_list_different_graph)
    for single_parallel in parallel_list:
        all_record_list.append(single_parallel[1])
        print('network', single_parallel, 'time', time.strftime('%Y-%m-%d %H:%M:%S'))
    pool.close()
    return all_record_list


def plot_network_comparison_with_time():
    run_time = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    plot_save_dir = os.path.join(plot_dir, run_time)
    if mode != 'hpc_to_local':
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)

    config_src = 'Config_Fig7b.ini'
    config_dst = os.path.join(plot_save_dir, run_time + "-" + config_src)

    save_name = "Record_list"
    save_path = os.path.join(res_dir, save_name)

    all_record_list = []
    if mode == 'hpc':
        all_record_list = run_simulation_compris_network_show_all_process()
        print('done simulation')

        # Save simulation result
        with open(save_path, 'wb') as f:
            pickle.dump(all_record_list, f)
        # Save Config file
        shutil.copyfile(config_src, config_dst)

        # record the config file path to get plot
        with open("last_save_dir.txt", "w") as f:
            f.write(plot_save_dir + "\n")

    if mode == 'hpc_to_local':
        f = open(save_path, 'rb')
        all_record_list = pickle.load(f)

        with open("last_save_dir.txt", "r") as f:
            last_runtime = f.readline()[:-1]

        strategy_list_with_time_about_param = []
        end_fraction = []
        for index, record in enumerate(all_record_list):
            list_strategy = record.get_total_list_strategy_fraction()
            strategy_list_with_time = [[] for i in range(2)]
            for i in range(len(list_strategy)):
                for j in range(len(strategy_list_with_time)):
                    strategy_list_with_time[j].append(list_strategy[i][j])
            strategy_list_with_time_about_param.append(strategy_list_with_time)
            end_fraction.append(record.get_total_list_strategy_fraction()[-1])
        Nr = 1
        Nc = 1
        fig_1 = plt.figure(num='fig_1', figsize=(6, 6))
        fig_1_axes = fig_1.subplots(Nr, Nc, squeeze=False)

        for i in range(Nr):
            for j in range(Nc):
                fig_1_axes[i, j].set_ylim(-0.05, 1.05)
                fig_1_axes[i, j].set_xlim(
                    0,
                    len(strategy_list_with_time_about_param[0][0]) +
                    1)
                fig_1_axes[i, j].set_xscale('symlog', linthresh=10)
                fig_1_axes[i, j].tick_params(axis='x', labelsize=18)
                fig_1_axes[i, j].tick_params(axis='y', labelsize=18)
                fig_1_axes[i][j].set_yticks(np.linspace(0, 1, 5).round(decimals=2))
                fig_1_axes[i][j].set_yticklabels(np.linspace(0, 1, 5).round(decimals=2),
                                                 fontsize=18)

                for k in range(len(strategy_list_with_time_about_param)):
                    plot_value = strategy_list_with_time_about_param[k][0]
                    fig_1_axes[i][j].plot(plot_value,
                                          color=color_set[k],
                                          linewidth=2)
                labels = ['Regular Graph', 'Erdos-Renyi Graph', 'Small-World Graph', 'Barabási-Albert Graph']
                fig_1_axes[i, j].legend(labels=labels, loc='upper left', fontsize=12)
                fig_1_axes[i, j].set_xlabel('time', fontsize=18)
                fig_1_axes[i, j].set_ylabel(r'$\%s$' % 'rho_{C}', fontsize=18)

        if mode == "hpc_to_local":
            with open("last_save_dir.txt", "r") as f:
                last_runtime = f.readline()[:-1]
        else:
            last_runtime = plot_save_dir

        fig_f_name = os.path.join(last_runtime, 'f-' + run_time + '.svg')
        fig_1.savefig(fig_f_name)
        plt.show(block=True)
        print('plot save')


def main():
    if switch_fuc == 'f':
        plot_network_comparison_with_time()


if __name__ == "__main__":
    main()
