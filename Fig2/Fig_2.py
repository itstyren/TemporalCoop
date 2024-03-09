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
conf.read('./Config_Fig2.ini', encoding='UTF-8')

# Set parameter value
agent_num = int(conf.get('network', 'agent_num'))
simulation_time = int(conf.get('network', 'simulation_time'))
initial_ratio = eval(conf.get('param', 'initial_ratio'))
num_processes = int(conf.get('pool', 'num_processes'))

# Set plot function
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


# plot Fig_2
def plot_colormap_compare_four_param(limit_iterator_list_1,
                                    limit_iterator_list_2,
                                    unlimit_iterator_list_1,
                                    unlimit_iterator_list_2,
                                    list_name_1, list_name_2, list_name_3, list_name_4):

    run_time = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    plot_save_dir = os.path.join(plot_dir, run_time)
    if mode != 'hpc_to_local':
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)

    config_src = 'Config_Fig2.ini'
    config_dst = os.path.join(plot_save_dir, run_time + "-" + config_src)

    save_name = "strategy_destination_compare_four_param"
    save_path = os.path.join(res_dir, save_name)

    strategy_destination = []
    if mode == 'hpc':
        graph = toolbox.init_regular_network(agent_num)
        # Run simulation
        strategy_destination = run_simulation_show_destination_compare_four_param_unroll(graph,
                                                                                        limit_iterator_list_1,
                                                                                        limit_iterator_list_2,
                                                                                        unlimit_iterator_list_1,
                                                                                        unlimit_iterator_list_2,
                                                                                        list_name_1.split('_')[
                                                                                                0],
                                                                                        list_name_2.split('_')[
                                                                                                0],
                                                                                        list_name_3.split('_')[
                                                                                                0],
                                                                                        list_name_4.split('_')[
                                                                                                0]
                                                                                        )
        # Save simulation result
        start_t = datetime.datetime.now()
        with open(save_path, 'wb') as f:
            pickle.dump(strategy_destination, f)
        end_t = datetime.datetime.now()
        elapsed_sec = (end_t - start_t).total_seconds()
        shutil.copyfile(config_src, config_dst)
        print('save time:', datetime.timedelta(seconds=elapsed_sec))
        # Record the current running time
        with open("../last_save_dir.txt", "w") as f:
            f.write(plot_save_dir + "\n")

    # Read simulation result
    if mode == 'hpc_to_local':
        with open(save_path, 'rb') as f:
            strategy_destination = pickle.load(f)

        # Set figure size
        Nr = len(limit_iterator_list_1)    # subfig row
        Nc = len(limit_iterator_list_2)    # subfig column
        if Nr == 2:
            fig_1 = plt.figure(num='fig_1', figsize=(15, 10))
        elif Nc == 1:
            fig_1 = plt.figure(num='fig_1', figsize=(8, 8))
        else:
            fig_1 = plt.figure(num='fig_1', figsize=(11, 6))

        fig_1.subplots_adjust(
            hspace=0.15,
            wspace=0.2,
        )
        fig_1_axes = fig_1.subplots(Nr, Nc, squeeze=False)

        # Get plot data
        graph_list_with_cooperation_fraction = []
        for graph in strategy_destination:
            y_list = []
            for y_value in graph:
                x_list = []
                for x_value in y_value:
                    x_list.append(
                        np.around(x_value[0], decimals=2))
                y_list.append(x_list)
            y_list = y_list[::-1]
            graph_list_with_cooperation_fraction.append(y_list)
        images = []
        list_index = 0

        # plot colormap
        for i in range(Nr):
            for j in range(Nc):
                images.append(fig_1_axes[i, j].imshow(
                    gaussian_filter(
                        graph_list_with_cooperation_fraction[list_index],
                        sigma=1.0),
                    cmap=cm.RdYlBu,
                    interpolation='quadric',
                    aspect="auto"))
                fig_1_axes[i, j].set_xlim(0, len(unlimit_iterator_list_2) - 1)
                if i == 0:
                    fig_1_axes[i, j].set_ylim(
                        len(unlimit_iterator_list_1) - 1, 0)
                    fig_1_axes[i, j].set_yticks(
                        np.linspace(0,
                                    len(unlimit_iterator_list_1) - 1, 3))
                    fig_1_axes[i, j].set_yticklabels(np.flipud(
                        np.around(np.linspace(unlimit_iterator_list_1[0],
                                              unlimit_iterator_list_1[-1], 3),
                                  decimals=2)),
                        fontsize=18)
                    fig_1_axes[i, j].set_title('%s=%s' %
                                               (list_name_2.split('_')[0],
                                                str(limit_iterator_list_2[j])))
                    fig_1_axes[i, j].set_ylabel('%s' %
                                                (list_name_3.split('_')[0]),
                                                rotation=0,
                                                fontsize=18)
                else:
                    fig_1_axes[i, j].set_ylim(9, 0)
                    fig_1_axes[i, j].set_yticks(np.linspace(0, 9, 4))
                fig_1_axes[i, j].set_xticks(
                    np.linspace(0,
                                len(unlimit_iterator_list_2) - 1, 3))
                fig_1_axes[i, j].set_xticklabels(np.around(np.linspace(
                    unlimit_iterator_list_2[0],
                    unlimit_iterator_list_2[-1], 3),
                    decimals=2),
                    fontsize=18)
                fig_1_axes[i, j].set_xlabel(r'r',
                                            fontsize=18)
                fig_1_axes[i, j].label_outer()
                if Nr == 1:

                    if j == len(limit_iterator_list_2) - 1:
                        fig_1_axes[i, j].text(
                            len(unlimit_iterator_list_2),
                            len(unlimit_iterator_list_2) / 2,

                            '%s=%s' % (list_name_1.split('_')[0],
                                       str(limit_iterator_list_1[i])),
                            verticalalignment='center',
                            rotation=0)
                list_index += 1

        norm = colors.Normalize(vmin=0, vmax=1)
        for im in images:
            im.set_norm(norm)

        def update(changed_image):
            for im in images:
                if (changed_image.get_cmap() != im.get_cmap()
                        or changed_image.get_clim() != im.get_clim()):
                    im.set_cmap(changed_image.get_cmap())
                    im.set_clim(changed_image.get_clim())

        for im in images:
            im.callbacks.connect('changed', update)
        if Nr == 1 and Nc != 1:
            cbar = fig_1.colorbar(images[0],
                                  ax=fig_1_axes,
                                  orientation='horizontal',
                                  fraction=0.05,
                                  shrink=0.8,
                                  aspect=20)
        elif Nc == 1:
            cbar = fig_1.colorbar(images[0],
                                  ax=fig_1_axes,
                                  fraction=0.05,
                                  shrink=1,
                                  aspect=20)
        else:
            cbar = fig_1.colorbar(images[0],
                                  ax=fig_1_axes,
                                  fraction=0.05,
                                  shrink=1,
                                  aspect=20)

        last_runtime = plot_save_dir

        fig_2_name = os.path.join(last_runtime, 'fig2-' + run_time + '.svg')
        fig_1.savefig(fig_2_name)
        plt.show()
        print('plot save')


# Simulation process of evolutionary dynamics
def set_init_agent_strategy(agent_list):
    for i in range(len(agent_list)):
        # Cooperator is 0, Defector is 1
        strategy = np.random.choice([0, 1], p=initial_ratio.ravel())
        agent_list[i].set_init_strategy(strategy)

def run_simulation_show_destination_compare_four_param_unroll(graph,
                                                       limit_iterator_list_1, limit_iterator_list_2,
                                                       unlimit_lterator_list_1, unlimit_lterator_list_2,
                                                       param_name_1, param_name_2, param_name_3, param_name_4):
    # Number of parallel processes
    pool = mp.Pool(num_processes)

    agent_list = []
    # Set iterable agent list according to generated graph
    for i, node in enumerate(graph):
        agent_list.append(Agent(i, node, graph))

    # Randomly assign strategy with initial_ratio
    set_init_agent_strategy(agent_list)

    strategy_destination_all_graph = []

    # Parameter iteration
    for limit_iterator_1_index, limit_iterator_1_value in enumerate(
            limit_iterator_list_1):

        for limit_iterator_2_index, limit_iterator_2_value in enumerate(
                limit_iterator_list_2):
            strategy_destination_each_sub_graph = []
            param_dict = {
                param_name_1: limit_iterator_1_value,
                param_name_2: limit_iterator_2_value,
            }
            # Generate an iteration tuple list according to the iteration value
            unlimit_iterator_list_tuple = []
            for unlimit_iterator_index, unlimit_iterator_value in enumerate(unlimit_lterator_list_1):
                for unlimit_iterator_index_1, unlimit_iterator_value_2 in enumerate(unlimit_lterator_list_2):
                    unlimit_iterator_list_tuple.append((unlimit_iterator_value, unlimit_iterator_value_2))
            # Shuffle the order and balance the load
            random.shuffle(unlimit_iterator_list_tuple)

            param_name_tuple = (param_name_3, param_name_4)

            # Parallel computing
            partial_func = partial(
                parallel_all_process_and_destination,
                param_dict, param_name_tuple, agent_list)
            unordered_res = pool.map_async(partial_func, unlimit_iterator_list_tuple)
            unordered_res.wait()
            # Get results list
            unordered_res = unordered_res.get()
            for u_i_1 in unlimit_lterator_list_1:
                single_strategy_destination = []
                for u_i_2 in unlimit_lterator_list_2:
                    for i, res in enumerate(unordered_res):
                        if (u_i_1, u_i_2) == res[1]:
                            single_strategy_destination.append(res[0])
                strategy_destination_each_sub_graph.append(single_strategy_destination)
            strategy_destination_all_graph.append(
                strategy_destination_each_sub_graph)
    pool.close()
    pool.join()
    return strategy_destination_all_graph


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
    if switch_fuc == 'c':
        plot_colormap_compare_four_param(
                k_iterator_list,
                sigma_iterator_list,
                p_iterator_list,
                r_iterator_list,
                toolbox.retrieve_name(k_iterator_list),
                toolbox.retrieve_name(sigma_iterator_list),
                toolbox.retrieve_name(p_iterator_list),
                toolbox.retrieve_name(r_iterator_list))


if __name__ == "__main__":
    main()