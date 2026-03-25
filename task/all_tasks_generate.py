from matAgent.ccpso_50d import FiftyDimCCPsoSwarm
from matAgent.clpso import ClpsoSwarm
from matAgent.pso import PsoSwarm
from matAgent.rlepso import RlepsoSwarm
from matAgent.rl_ccpso_eval import RlCCPsoSwarm
from matAgent.testpso import TestpsoSwarm


# def all_tasks_generate():
#     baseline_optimizers = [
#         PsoSwarm,
#         ClpsoSwarm,
#     ]
#     rl_optimizer_pairs = [
#         {
#             'train_optimizer': TestpsoSwarm,
#             'evaluate_optimizer': RlepsoSwarm,
#             'train_profile': 'original_rlepso',
#             'train_al_type': 'testpso',
#         },
#         {
#             'train_optimizer': FiftyDimCCPsoSwarm,
#             'evaluate_optimizer': RlCCPsoSwarm,
#             'train_profile': 'original_rlepso',
#             'train_al_type': 'ccpso_50d',
#         },
#     ]
#
#     task = {
#         'type': 'top',
#         'baseline_optimizers': baseline_optimizers,
#         'rl_optimizer_pairs': rl_optimizer_pairs,
#         # 'evaluate_function': list(range(1, 29)),
#         'evaluate_function': [1, 15, 23],
#         'runtimes': 5,
#         # 'separate_trains': [False],
#         'separate_trains': [True],
#         'groups': [5],
#         'train_max_episode': 200,
#         'train_max_steps': 8000,
#         'dims': [50],
#         'train_times': 1,
#         'max_fe': int(1e4),
#         'n_part': 100,
#         'lr_critic': 1e-7,
#         'lr_actor': 1e-9,
#     }
#
#     return [task]
#
#
# if __name__ == '__main__':
#     print(all_tasks_generate())
def all_tasks_generate():
    baseline_optimizers = [] # 清空基线算法，节省时间，只跑 RL 对比
    rl_optimizer_pairs = [
        {
            'train_optimizer': TestpsoSwarm,
            'evaluate_optimizer': RlepsoSwarm,
            'train_profile': 'original_rlepso',
            'train_al_type': 'testpso',
        },
        {
            'train_optimizer': FiftyDimCCPsoSwarm,
            'evaluate_optimizer': RlCCPsoSwarm,
            'train_profile': 'original_rlepso',
            'train_al_type': 'ccpso_50d',
        },
    ]

    task = {
        'type': 'top',
        'baseline_optimizers': baseline_optimizers,
        'rl_optimizer_pairs': rl_optimizer_pairs,
        'evaluate_function': [1],      # 极速验证：只跑 F1 函数
        'runtimes': 1,                 # 极速验证：每种算法只独立运行 1 次（不求均值了）
        'separate_trains': [False],    # 不为单独函数分离训练
        'groups': [5],
        'train_max_episode': 2,        # 极速验证：只跑 2 个 episode 意思一下
        'train_max_steps': 500,        # 极速验证：你要求的 500 轮
        'dims': [30],                  # 极速验证：你要求的 30 维度
        'train_times': 1,
        'max_fe': 500,                 # 评估阶段也只给 500 次评价机会
        'n_part': 50,                  # 减少粒子数到 50 加快单步速度
        'lr_critic': 1e-3,             # PyTorch 推荐的正常学习率
        'lr_actor': 1e-4,
    }
    return [task]

if __name__ == '__main__':
    freeze_support()
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    res = 'restart'
    processes_count = 2  # 你要求的 2 个线程并发
    while res == 'restart':
        res = main(processes_count)
        logger.info(f'main run finish res:{res}')
        time.sleep(60)