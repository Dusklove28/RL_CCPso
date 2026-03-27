from log import logger
from task_run_utils.top_task_run import main

# 注意：导入路径已根据你的重命名重构 (ccpso.py 和 ccpso_eval.py)
from matAgent.ccpso import FiftyDimCCPsoSwarm  # 如果类名在重构时也改了，请同步修改为对应的类名 (如 CCPsoSwarm)
from matAgent.testpso import TestpsoSwarm
from matAgent.rlepso import RlepsoSwarm
from matAgent.ccpso_eval import RlCCPsoSwarm

import multiprocessing as mp
from multiprocessing import freeze_support
import time

def all_tasks_generate():
    task = {
        'type': 'top',
        'baseline_optimizers': [],  # 【满足要求】清空基线算法，不跑传统 PSO 算法对比
        'rl_optimizer_pairs': [
            {
                # 算法一：rlpso (原作者)
                'train_optimizer': TestpsoSwarm,
                'evaluate_optimizer': RlepsoSwarm,
                'train_profile': 'original_rlepso',
                'train_al_type': 'testpso',
            },
            {
                # 算法二：rlccpso (你的)
                'train_optimizer': FiftyDimCCPsoSwarm,  # 需与重构后的类名一致
                'evaluate_optimizer': RlCCPsoSwarm,
                'train_profile': 'original_rlepso',
                'train_al_type': 'ccpso',  # 配合你的文件重命名，内部标识名也建议改为 ccpso
            },
        ],
        'evaluate_function': [1,],  # 【满足要求】只运行 1, 15, 23 三个测试函数
        'runtimes': 5,  # 【关键恢复】评估阶段运行次数。为了消除随机性画出平滑的收敛对比图，建议设为 10 或以上
        'separate_trains': [True],  # 【关键恢复】针对每个函数单独训练，对应你之前的实验逻辑
        'groups': [5],
        'train_max_episode': 200,  # 【关键恢复】恢复为 200 次训练迭代
        'train_max_steps': 8000,
        'dims': [30],  # 【关键恢复】恢复为你要求的 30 维度
        'train_times': 1,
        'max_fe': int(1e4),  # 【关键恢复】正常的适应度评估上限 (10000次)
        'n_part': 100,  # 恢复正常的种群大小
        'lr_critic': 1e-4,
        'lr_actor': 1e-6,
    }

    return [task]


if __name__ == '__main__':
    freeze_support()
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    res = 'restart'
    processes_count = 2  # 【满足要求】严格限制为 2 个线程并发

    while res == 'restart':
        res = main(processes_count)
        # logger.info(f'main run finish res:{res}')
        print(f'main run finish res:{res}')
        time.sleep(60)