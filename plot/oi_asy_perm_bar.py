from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import os
from typing import List, Dict, Tuple
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

import pandas as pd

plt.rcParams['axes.facecolor'] = '#f5f6fa'

RESULT_PATH = '/home/PJLAB/kang/proj/graph_offline_imitation/results/'

FIGNAME     = 'Adroit_temp'

ALL_ENV = [
    # "AntMaze-ExpDiv-10",
    # 'AntMaze-ExpDiv-3',
    # 'AntMaze-ExpDiv-1',
    
    'Adroit-Pen-Human-3',
]

env2exp = {
    # 'AntMaze-ExpDiv-10':        'oi_antmaze_umaze_exp_div_10',
    # 'AntMaze-ExpDiv-3':         'oi_antmaze_umaze_exp_div_3',
    # 'AntMaze-ExpDiv-1':         'oi_antmaze_umaze_exp_div_1',

    'Adroit-Pen-Human-3':       'oi_adroit_pen_human_3',
}

alg2exp = {
    'BC-all':                   'bc_all',
    'ORIL':                     'oril',
    # 'DWBC':                     'dwbc',
    # 'SMODICE':                  'smodice',
    'ContrastiveOI-V2':         'contrastiveoi_v2',
}

alg2color = {
    'BC-all':                   'green',
    'ORIL':                     'grey',
    'ContrastiveOI-V2':         'darkorange',
    'DWBC':                     'cornflowerblue',
    'SMODICE':                  '#23191a'
}


all_seeds = [
    10,
    11,
    # 12,
    # 13,
    # 14,
]


def plot_perf_bars():
    fig, ax  = plt.subplots(1, 1, figsize=((len(ALL_ENV) + 2) * 1.25, 5), tight_layout=True,)

    width    = 0.1
    ax.set_xlim(0, 2.5)
    x_tick_pos  =   np.array([0.75 * i_env + 0.4 + width for i_env in range(len(ALL_ENV))])
    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(ALL_ENV, rotation=60, ha='right')
    initial_x_ticks = x_tick_pos - width * 2
    
    all_algs = list(alg2exp.keys())
    for i_alg, alg in enumerate(all_algs):
            
        score_mean = []
        score_std  = []

        for j in range(len(ALL_ENV)):
            
            env                 = ALL_ENV[j]
            scores_across_seeds = []
            
            seeds = all_seeds
            for seed in seeds:
                exp_path = RESULT_PATH + f"{env2exp[env]}/" + f'{alg2exp[alg]}-{seed}/'

                with open(exp_path + 'log.csv', 'rb') as f:
                    log = pd.read_csv(f)

                # rewards = log['eval/reward']
                rewards = log['eval/score']
                
                scores_across_seeds.append(np.max(rewards))

            score_mean.append(np.mean(scores_across_seeds))
            score_std.append(np.std(scores_across_seeds) * 0.95 / 2)

            print(f"{env} - {alg} : {score_mean[-1]} ({score_std[-1]})")

        ax.bar(
            initial_x_ticks + width * i_alg, 
            score_mean, 
            width       =   width, 
            color       =   alg2color[alg],
            edgecolor   =   'white', 
            linewidth   =   0.1, 
            yerr=score_std, 
            capsize=3, 
            label=alg
        )


    ax.legend()

    sns.despine(ax=ax)
    # ax.grid(linestyle='--')
    # ax.set_title(f'{env}', fontsize=16)
    
    ax.set_ylabel('Score', fontsize=15)
    ax.set_xlabel('', fontsize=15)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)

    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(12) 
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(12) 

    fig.savefig(f"{Path(__file__).resolve().parent}/{FIGNAME}.png")

    plt.show()

    



if __name__ == '__main__':
    plot_perf_bars()