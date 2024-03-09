import configparser, pickle, datetime, os, shutil, time
import matplotlib.pyplot as plt
from matplotlib import colors
from functools import partial
import multiprocessing as mp
from class_record import *
import tools as toolbox

# Read config
conf = configparser.ConfigParser()
conf.read('./Config_Fig6.ini', encoding='UTF-8')

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
# the number of snapshots
simulation_time_break_num = int(conf.get('plot', 'simulation_time_break_num'))
time_set = [35000, 47500]

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

    # set param_name as global variable
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

        # record all agents's state
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
          's\ttime', time.strftime('%Y-%m-%d %H:%M:%S'))

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


def run_simulation_show_all_process(graph, single_iterator_list, param_name):
    # Number of parallel processes
    pool = mp.Pool(num_processes)

    agent_list = []
    all_record_list = []

    # Set iterable agent list according to generated graph
    for i, node in enumerate(graph):
        agent_list.append(Agent(i, node, graph))

    # Randomly assign strategy with initial_ratio
    set_init_agent_strategy(agent_list)

    param_dict = {}

    partial_func = partial(parallel_all_process_and_destination,
                           param_dict, param_name, agent_list)
    parallel_list = pool.map(partial_func, single_iterator_list)

    for single_parallel in parallel_list:
        all_record_list.append(single_parallel[1])
        print(param_name, single_parallel, 'time', time.strftime('%Y-%m-%d %H:%M:%S'))
    pool.close()

    return all_record_list


def plot_evolutionary_processes_with_one_param(single_iterator_list,
                                               list_name):
    run_time = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    plot_save_dir = os.path.join(plot_dir, run_time)
    if mode != 'hpc_to_local':
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)

    config_src = 'Config_Fig6.ini'
    config_dst = os.path.join(plot_save_dir, run_time + "-" + config_src)

    save_name = "Record_list"
    save_path = os.path.join(res_dir, save_name)

    all_record_list = []
    if mode == 'hpc':
        graph = toolbox.init_regular_network(agent_num)
        # Run simulation
        all_record_list = run_simulation_show_all_process(
            graph, single_iterator_list,
            list_name.split('_')[0])
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

        # Plot Fig.6 (*-1)
        if switch_fuc == 'a':
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
            Nc = len(single_iterator_list)
            fig_1 = plt.figure(num='fig_1', figsize=(8, 6))
            fig_1.subplots_adjust(
                hspace=0.3,
                wspace=0.2,
            )
            fig_2 = plt.figure(num='fig_2', figsize=(8, 8))
            fig_2.subplots_adjust(
                hspace=0.3,
                wspace=0.2,
            )
            labels_1 = ['C', 'D']
            fig_1_axes = fig_1.subplots(Nr, Nc, squeeze=False)
            fig_2_axes = fig_2.subplots(Nr, Nc, squeeze=False)

            list_index = 0
            cmap = colors.ListedColormap(
                [color_set[0], color_set[1], color_set[2], color_set[3]])
            pie_color = color_set[0:3]
            bounds = [0, 1, 2, 3, 4]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            explode = (0, 0.1)
            for i in range(Nr):
                for j in range(Nc):
                    fig_1_axes[i, j].set_ylim(-0.05, 1.05)
                    fig_1_axes[i, j].set_xlim(0, len(strategy_list_with_time_about_param[list_index][0]) + 1)
                    fig_1_axes[i, j].set_xscale('symlog', linthresh=10)
                    fig_1_axes[i, j].tick_params(axis='x', labelsize=16)
                    fig_1_axes[i, j].tick_params(axis='y', labelsize=16)

                    for strategy_index in range(
                            len(strategy_list_with_time_about_param[0])):
                        fig_1_axes[i][j].plot(
                            strategy_list_with_time_about_param[list_index]
                            [strategy_index],
                            color=color_set[strategy_index])

                    fig_1_axes[i][j].set_title('%s=%s' %
                                               (list_name.split('_')[0],
                                                single_iterator_list[list_index]), fontsize=24)
                    fig_2_axes[i][j].set_title('%s=%s' %
                                               (list_name.split('_')[0],
                                                single_iterator_list[list_index]))
                    fig_1_axes[i, j].set_xlabel('time', fontsize=20)
                    fig_1_axes[i, j].set_ylabel('strategy fraction', fontsize=20)
                    fig_1_axes[i, j].label_outer()
                    labels = [
                        n if v > 0 else '' for n, v in zip([
                            '{:.2f}%'.format(i * 100)
                            for i in end_fraction[list_index]
                        ], end_fraction[list_index])
                    ]

                    fig_2_axes[i, j].pie(
                        end_fraction[list_index],
                        labels=labels,
                        explode=explode,
                        shadow=True,
                        startangle=90,
                        colors=pie_color)
                    list_index += 1

            fig_1.legend(labels=labels_1,
                         handlelength=12,
                         handleheight=0.1,
                         loc='center',
                         ncol=4,
                         fontsize=12,
                         columnspacing=4,
                         frameon=False,
                         bbox_to_anchor=(0.51, 0.93, 0, 0))

            fig_a1_name = os.path.join(last_runtime, 'a1-' + run_time + '.svg')
            fig_a2_name = os.path.join(last_runtime, 'a2-' + run_time + '.svg')
            fig_1.savefig(fig_a1_name)
            fig_2.savefig(fig_a2_name)
            plt.show(block=True)
            print('plot save')

        # Plot Fig.6 (*-2) and (*-3)
        elif switch_fuc == 'b':
            all_record_list = all_record_list[0]    # subfig (a-*)
            strategy_data = []
            state_data = []
            for t in time_set:
                strategy_data.append(np.array(all_record_list.get_agent_strategy_list()[t]))
                state_data.append(np.array(all_record_list.get_agent_state_list()[t]))
            data_set = []
            # AC=1,AD=2,IC=3,ID=4
            for i in range(len(time_set)):
                data_set.append(np.array([0 for _ in range(agent_num)]))
                for j in range(agent_num):
                    if state_data[i][j] == 1 and strategy_data[i][j] == 0:
                        data_set[i][j] = 1
                    if state_data[i][j] == 1 and strategy_data[i][j] == 1:
                        data_set[i][j] = 2
                    if state_data[i][j] == 0 and strategy_data[i][j] == 0:
                        data_set[i][j] = 3
                    if state_data[i][j] == 0 and strategy_data[i][j] == 1:
                        data_set[i][j] = 4
                data_set[i] = np.array(np.split(data_set[i], np.sqrt(agent_num), axis=0))

            Nr = 1
            Nc = len(time_set)
            fig_1 = plt.figure(num='fig_1', figsize=(10, 5))
            fig_1.subplots_adjust(hspace=0.1, wspace=0.1)
            fig_1_axes = fig_1.subplots(Nr, Nc, squeeze=False)
            b_color = colors.ListedColormap(['blue', 'red', 'lightsteelblue', 'lightcoral'])
            # Set the value corresponding to the color
            b_norm = colors.Normalize(vmin=1, vmax=4)

            subplot_index = 0
            for i in range(Nr):
                for j in range(Nc):
                    # The matrix is inverted, since pcolor is drawn backwards from the last row
                    subplot_data = np.flipud(data_set[subplot_index])
                    fig_1_axes[i][j].pcolor(subplot_data, cmap=b_color, norm=b_norm)
                    fig_1_axes[i][j].axis('off')
                    fig_1_axes[i][j].axis('square')
                    fig_1_axes[i][j].set_title('%s=%s' % ('t', time_set[subplot_index]), fontsize=18)
                    subplot_index += 1
            fig_1.tight_layout()

            fig_b_name = os.path.join(last_runtime, 'b-' + run_time + '.svg')
            fig_1.savefig(fig_b_name)
            plt.show(block=True)
            print('plot save')
        else:
            raise ValueError("Invalid switch_fuc. switch_fuc must be a or b")


def main():
    if switch_fuc == 'a':
        plot_evolutionary_processes_with_one_param(
            r_iterator_list, toolbox.retrieve_name(r_iterator_list))
    elif switch_fuc == 'b':
        plot_evolutionary_processes_with_one_param(
            r_iterator_list, toolbox.retrieve_name(r_iterator_list))
    else:
        raise ValueError("Invalid switch_fuc. switch_fuc must be a or b")


if __name__ == "__main__":
    main()
