import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

sns.set(font_scale=1.3)


def values(df, directory):
    all_df = df.loc[df['type'] == 'all']

    plt.figure('joint_values')
    sns.tsplot(data=all_df, time='step', value='mean_value', unit='run', condition='n_share')
    plt.xlabel('step')
    plt.ylabel('global return')
    plt.tight_layout()
    plt.savefig(directory + '/joint_values.png')


def schelling(df, directory):
    final_df = df.loc[df['step'] == df['step'].max()]
    final_df = final_df.loc[final_df['type'] != 'all']
    final_df = final_df.groupby(['type', 'n_share', 'run']).mean()
    final_df = final_df.reset_index()
    final_df.to_csv(directory + '/schelling.csv', sep='\t', encoding='utf-8')

    plt.figure('schelling')
    sns.boxplot(data=final_df, x='n_share', y='mean_value', hue='type')
    plt.xlabel('number of sharing agents')
    plt.ylabel('mean individual return')
    plt.tight_layout()
    plt.savefig(directory + '/schelling.png')

    plt.figure('schelling log')
    sns.boxplot(data=final_df, x='n_share', y='mean_value', hue='type')
    plt.xlabel('number of sharing agents')
    plt.ylabel('mean individual return')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(directory + '/schelling_log.png')

    plt.figure('schelling ts')
    sns.tsplot(data=final_df, time='n_share', value='mean_value', unit='run', condition='type')
    plt.xlabel('number of sharing agents')
    plt.ylabel('mean individual return')
    plt.tight_layout()
    plt.savefig(directory + '/schelling_ts.png')


def trades(df, directory):
    df = df.groupby(['n_share', 'step']).mean()
    df = df.reset_index()

    plt.figure('trades')
    sns.tsplot(data=df, time='step', value='mean_trade', unit='run', condition='n_share', ci='sd')
    plt.xlabel('step')
    plt.ylabel('mean shared value')
    plt.tight_layout()
    plt.savefig(directory + '/trades.png')


def plot(directory):
    df = pd.read_csv(directory + '/values.csv', sep='\t', encoding='utf-8', index_col=0)
    values(df, directory)
    schelling(df, directory)

    df = pd.read_csv(directory + '/trades.csv', sep='\t', encoding='utf-8', index_col=0)
    trades(df, directory)


'''
for folder in ['simple_10', 'simple_50', 'market_10', 'market_50']:
    plot('./results/' + folder)
    plt.close("all")
'''