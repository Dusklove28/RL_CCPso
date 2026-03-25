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
        'baseline_optimizers': [],  # 不跑传统算法对比
        'rl_optimizer_pairs': [  # 只留一对 RL 算法验证
            {
                'train_optimizer': TestpsoSwarm,
                'evaluate_optimizer': RlepsoSwarm,
                'train_profile': 'original_rlepso',
                'train_al_type': 'testpso',
            }
        ],
        'evaluate_function': [1],  # 只测试 F1
        'runtimes': 1,  # 评估阶段只运行 1 次（不求均值）
        'separate_trains': [False],
        'groups': [1],  # 群组数降到 1
        'train_max_episode': 1,  # 【核心】只训练 1 局！
        'train_max_steps': 10,  # 【核心】这局只跑 10 步就结束！
        'dims': [2],  # 【核心】把 30 维降到 2 维，计算量锐减！
        'train_times': 1,  # 只训练 1 个模型
        'max_fe': 50,  # 【核心】评估阶段，最多算 50 次适应度就强制结束！
        'n_part': 10,  # 【核心】整个种群只放 10 个粒子！
        'lr_critic': 1e-3,
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
#########为什么还是全流程，
        # # 生成全流程任务
        # from matAgent.clpso import ClpsoSwarm
        # from matAgent.epso import EpsoSwarm
        # from matAgent.fdrpso import FdrpsoSwarm
        # from matAgent.hpso_tvac import HpsotvacSwarm
        # from matAgent.lips import LipsSwarm
        # from matAgent.olpso import OlpsoSwarm
        # from matAgent.pso import PsoSwarm
        # from matAgent.shpso import ShpsoSwarm
        # from matAgent.hrlepso_base import HrlepsoBaseSwarm
        # from matAgent.swarm.gwo import GwoSwarm
        # from matAgent.rlepso import RlepsoSwarm
        #
        #
        # def test_all_tasks_generate():
        #     evaluate_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, PsoSwarm, ShpsoSwarm,
        #                            HrlepsoBaseSwarm,
        #                            GwoSwarm]
        #     base_evaluate_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, OlpsoSwarm, PsoSwarm,
        #                                 ShpsoSwarm,
        #                                 EpsoSwarm, GwoSwarm]  # 都用一样的
        #     runtimes = 1
        #     separate_trains = [False]
        #     groups = [5, ]
        #     train_max_episode = 2
        #     train_max_steps = 100 * train_max_episode
        #     dims = [20, ]
        #     evaluate_function = list(range(1, 2, 1))
        #
        #     task = {'type': 'top',
        #             'evaluate_optimizers': evaluate_optimizers,
        #             'base_evaluate_optimizers': base_evaluate_optimizers,
        #             'evaluate_function': evaluate_function,
        #             'runtimes': runtimes,
        #             'separate_trains': separate_trains,
        #             'groups': groups,
        #             'train_max_episode': train_max_episode,
        #             'train_max_steps': train_max_steps,
        #             'dims': dims,
        #             'train_times': 1,
        #             'lr_critic': 1e-4,
        #             'lr_actor': 1e-6,
        #             }
        #
        #     return [task]
        #
        #
        # def all_tasks_generate():
        #     # evaluate_optimizers = [GwoSwarm]
        #     evaluate_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, PsoSwarm, ShpsoSwarm,
        #                            HrlepsoBaseSwarm]
        #     # evaluate_optimizers = [PsoSwarm]
        #     base_evaluate_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, OlpsoSwarm, PsoSwarm,
        #                                 ShpsoSwarm,
        #                                 EpsoSwarm, ]  # 都用一样的
        #     runtimes = 10
        #     separate_trains = [True, False]
        #     # groups = [1, 3, 5, 7, 9]
        #     groups = [5]
        #     train_max_episode = 400
        #     train_max_steps = train_max_episode * 100
        #     dims = [30, 50]
        #     evaluate_function = list(range(1, 29, 1))
        #
        #     task = {'type': 'top',
        #             'evaluate_optimizers': evaluate_optimizers,
        #             'base_evaluate_optimizers': base_evaluate_optimizers,
        #             'evaluate_function': evaluate_function,
        #             'runtimes': runtimes,
        #             'separate_trains': separate_trains,
        #             'groups': groups,
        #             'train_max_episode': train_max_episode,
        #             'train_max_steps': train_max_steps,
        #             'dims': dims,
        #             'train_times': 1,
        #             'lr_critic': 1e-4,
        #             'lr_actor': 1e-6,
        #             }
        #
        #     return [task]
        #
        #
        # def CLPSO_tasks_generate():
        #     evaluate_optimizers = [ClpsoSwarm, ]
        #     base_evaluate_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, OlpsoSwarm, PsoSwarm,
        #                                 ShpsoSwarm,
        #                                 EpsoSwarm]  # 都用一样的
        #     runtimes = 10
        #     # separate_trains = [True, False]
        #     separate_trains = [False]
        #     # groups = [1, 3, 5, 7, 9]
        #     groups = [5]
        #     train_max_episode = 600
        #     train_max_steps = train_max_episode * 100
        #     dims = [30]
        #     evaluate_function = list(range(1, 29, 1))
        #
        #     task = {'type': 'top',
        #             'evaluate_optimizers': evaluate_optimizers,
        #             'base_evaluate_optimizers': base_evaluate_optimizers,
        #             'evaluate_function': evaluate_function,
        #             'runtimes': runtimes,
        #             'separate_trains': separate_trains,
        #             'groups': groups,
        #             'train_max_episode': train_max_episode,
        #             'train_max_steps': train_max_steps,
        #             'dims': dims,
        #             'train_times': 1,
        #             'lr_critic': 1e-3,
        #             'lr_actor': 1e-5,
        #             }
        #
        #     return [task]
        #
        #
        # def QHrlepsoTrainTest():
        #     from matAgent.qrlepso.f16rlepso import F16Rlepso
        #     from matAgent.qrlepso.f64rlepso import F64Rlepso
        #     from matAgent.qrlepso.i8rlepso import I8Rlepso
        #     from matAgent.qrlepso.i16rlepso import I16Rlepso
        #     evaluate_optimizers = [F16Rlepso, F64Rlepso, I8Rlepso, I16Rlepso]
        #     # evaluate_optimizers = [I8Rlepso, ]
        #     base_evaluate_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, OlpsoSwarm, PsoSwarm,
        #                                 ShpsoSwarm,
        #                                 EpsoSwarm]  # 都用一样的
        #     runtimes = 10
        #     separate_trains = [False]
        #     groups = [5, ]
        #     train_max_episode = 200
        #     train_max_steps = 20000
        #     dims = [30]
        #     evaluate_function = list(range(1, 29, 1))
        #
        #     task = {'type': 'top',
        #             'evaluate_optimizers': evaluate_optimizers,
        #             'base_evaluate_optimizers': base_evaluate_optimizers,
        #             'evaluate_function': evaluate_function,
        #             'runtimes': runtimes,
        #             'separate_trains': separate_trains,
        #             'groups': groups,
        #             'train_max_episode': train_max_episode,
        #             'train_max_steps': train_max_steps,
        #             'dims': dims,
        #             'train_times': 1
        #             }
        #
        #     return [task]
        #
        #
        # if __name__ == '__main__':
        #     test_all_tasks_generate()
