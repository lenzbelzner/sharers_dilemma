import os
import datetime
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from dos.target_functions import *

sns.set()
result_dir = os.path.join(os.getcwd(), 'results/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(result_dir)

RUNS = 200
RANGE = 4
STEPS = 200
RESET_TRADE_STEPS = int(STEPS * 1.)
SAMPLES = 100
ALPHA = .5
ALPHA_SHARE = .5
PSI = .25
INIT_STD = 1.
MIN_STD = .2
MIN_STD_SHARE = .2
N_AGENTS = 4
NON_SHARING = 1

target_functions = {}
for i in range(N_AGENTS):
    target_functions[i] = generate_f(N_AGENTS)

value_log = []
trade_log = []

for run in tqdm(range(RUNS)):
    for i in range(N_AGENTS):
        target_functions[i] = generate_f(N_AGENTS)

    independent_means = {}
    independent_stds = {}
    for i in range(N_AGENTS):
        independent_means[i] = 0.
        independent_stds[i] = INIT_STD

    for step in range(STEPS):
        samples = {}
        for i in range(N_AGENTS):
            samples[i] = np.random.normal(independent_means[i], independent_stds[i], size=SAMPLES)

        actions = [samples[i] for i in range(N_AGENTS)]

        values = {}
        mean_vs = {}

        for agent in range(N_AGENTS):
            values[agent] = target_functions[agent](actions)
            v = values[agent]
            mean_vs[agent] = np.mean(v)
            # TODO: log min and mean value
            log = {'run': run, 'type': 'independent', 'step': step, 'mean_value': mean_vs[agent], 'agent': agent}
            value_log.append(log)

            sort = sorted(zip(v, samples[agent]))  # sorts ascending by value, low values first
            elite = sort[int(len(sort) * PSI):]  # elite selection

            elite_a = [a for _, a in elite]
            independent_means[agent] = (1. - ALPHA) * independent_means[agent] + ALPHA * np.mean(elite_a)
            independent_stds[agent] = (1. - ALPHA) * independent_stds[agent] + ALPHA * np.std(elite_a)
            independent_stds[agent] = max(independent_stds[agent], MIN_STD)

        value = np.sum([mean_vs[i] for i in range(N_AGENTS)])
        log = {'run': run, 'type': 'independent', 'step': step, 'mean_value': value, 'agent': 'joint'}
        value_log.append(log)

    action_means = {}
    action_stds = {}
    trade_means = {}
    trade_stds = {}

    for i in range(N_AGENTS):
        action_means[i] = 0.
        action_stds[i] = INIT_STD
        trade_means[i] = 0.
        trade_stds[i] = INIT_STD

    for step in range(STEPS):
        # reset sharing distribution every RESET_TRADE_STEPS
        if step % RESET_TRADE_STEPS == 0:
            for i in range(N_AGENTS):
                trade_means[i] = 0.
                trade_stds[i] = INIT_STD

        # sample from policy p_i for each agent
        samples = {}
        trades = {}
        for i in range(N_AGENTS):
            samples[i] = np.random.normal(action_means[i], action_stds[i], size=SAMPLES)
            trades[i] = np.random.normal(trade_means[i], trade_stds[i], size=SAMPLES)

        actions = [samples[i] for i in range(N_AGENTS)]

        # evaluate actions and clip trades
        values = {}
        for agent in range(N_AGENTS):
            values[agent] = target_functions[agent](actions)
            trades[agent] = np.clip(trades[agent], 0, values[agent])
            # TODO: Trade target softmax?
            # TODO: How to deal with changing optimal trade?

        # trade
        for i in range(N_AGENTS - NON_SHARING):
            values[i] = values[i] - trades[i] + np.mean(
                [trades[j] for j in trades if j != i and j < N_AGENTS - NON_SHARING])

        for i in range(N_AGENTS - NON_SHARING):
            log = {'run': run, 'type': 'sharing', 'step': step, 'mean_trade': np.mean(trades[i]), 'agent': i}
            trade_log.append(log)

        mean_vs = {}

        # update policy p_i for each agent
        for agent in range(N_AGENTS):
            v = values[agent]
            mean_vs[agent] = np.mean(v)
            # TODO: log min and mean value
            log = {'run': run, 'type': 'sharing', 'step': step, 'mean_value': mean_vs[agent], 'agent': agent}
            value_log.append(log)

            sort = sorted(zip(v, zip(samples[agent], trades[agent])))  # sorts ascending by value, low values first
            elite = sort[int(len(sort) * PSI):]

            elite_a = [a for _, (a, t) in elite]
            action_means[agent] = (1. - ALPHA) * action_means[agent] + ALPHA * np.mean(elite_a)
            action_stds[agent] = (1. - ALPHA) * action_stds[agent] + ALPHA * np.std(elite_a)
            action_stds[agent] = max(action_stds[agent], MIN_STD)

            elite_t = [t for _, (a, t) in elite]
            trade_means[agent] = (1. - ALPHA_SHARE) * trade_means[agent] + ALPHA_SHARE * np.mean(elite_t)
            trade_stds[agent] = (1. - ALPHA_SHARE) * trade_stds[agent] + ALPHA_SHARE * np.std(elite_t)
            trade_stds[agent] = max(trade_stds[agent], MIN_STD_SHARE)

        value = np.sum([mean_vs[i] for i in range(N_AGENTS)])
        log = {'run': run, 'type': 'sharing', 'step': step, 'mean_value': value, 'agent': 'joint'}
        value_log.append(log)

text_file = open(result_dir + '/seed.txt', 'w')
text_file.write(str(seed))
text_file.close()

df = pd.DataFrame(value_log)
df.to_csv(result_dir + '/values.csv', sep='\t', encoding='utf-8')

for agent in range(N_AGENTS):
    plt.figure(str(agent) + '_values')
    agent_df = df.loc[df['agent'] == agent]
    sns.tsplot(data=agent_df, time='step', value='mean_value', unit='run', condition='type')  # ci='sd'
    plt.savefig(result_dir + '/values_' + str(agent) + '.png')

df_base = df.loc[df['type'] == 'independent']
df_sharing = df.loc[df['type'] == 'sharing']
ratios = df_sharing['mean_value'].values / df_base['mean_value'].values
df_sharing['ratio'] = pd.Series(ratios, index=df_sharing.index)
df_sharing = df_sharing.loc[df_sharing['agent'] != 'joint']

plt.figure('ratios')
sns.tsplot(data=df_sharing, time='step', value='ratio', unit='run', condition='agent')
plt.savefig(result_dir + '/ratios.png')

plt.figure('ratios_median')
sns.tsplot(data=df_sharing, time='step', value='ratio', unit='run', condition='agent', estimator=np.median)
plt.savefig(result_dir + '/ratios_median.png')

plt.figure('joint_values')
joint_df = df.loc[df['agent'] == 'joint']
sns.tsplot(data=joint_df, time='step', value='mean_value', unit='run', condition='type')
plt.savefig(result_dir + '/joint_values.png')

df_base = joint_df.loc[joint_df['type'] == 'independent']
df_sharing = joint_df.loc[joint_df['type'] == 'sharing']
ratios = df_sharing['mean_value'].values / df_base['mean_value'].values
df_sharing.loc[:, 'ratio'] = ratios

plt.figure('joint_ratio')
sns.tsplot(data=df_sharing, time='step', value='ratio', unit='run')
plt.savefig(result_dir + '/joint_ratio.png')

plt.figure('joint_ratio_median')
sns.tsplot(data=df_sharing, time='step', value='ratio', unit='run', estimator=np.median)
plt.savefig(result_dir + '/joint_ratio_median.png')

plt.figure('trades')
df = pd.DataFrame(trade_log)
sns.tsplot(data=df, time='step', value='mean_trade', unit='run', condition='agent')
plt.savefig(result_dir + '/trades.png')

# TODO: Compare results for equal target functions with results for varying target functions
# TODO: Measure trading volume
# TODO: Measure effect of trading on individual rewards
# TODO: Measure effect of trading on disparity (e.g. gini index)
# TODO: Log maximum/minimum single agent value/utility (-> related to disparity)
# TODO: More than two agents, e.g. by increasing the trading action dimensionality.
# TODO: CMA-ES
# TODO: Effect of trade clipping/no clipping
