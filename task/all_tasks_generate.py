# 导入四个你要对比的算法
from matAgent.clpso import ClpsoSwarm
from matAgent.pso import PsoSwarm
from matAgent.rlepso import RlepsoSwarm  # 原作者包装好的 RL_TestPso
from matAgent.rl_ccpso_eval import RlCCPsoSwarm  # 你刚刚写的 RL_CCPso50D 包装类


def all_tasks_generate():
    # 1. 对比阵营：2个经典PSO，1个原版RL-PSO，1个你的RL-CCPso
    evaluate_optimizers = [PsoSwarm, ClpsoSwarm, RlepsoSwarm, RlCCPsoSwarm]
    base_evaluate_optimizers = []  # 如果原框架必须要求不为空，你可以填 [PsoSwarm]

    # 2. 粗验阶段：运行 5 次
    runtimes = 5

    # 3. 维度：【重要修正】你的 Env 里定义的是 50 维，这里必须也是 50 才能统一！
    dims = [50]

    # 4. 精选 4 个代表性测试函数（比如：1是单峰，20是多峰）
    evaluate_function = [1, 5, 10, 20]

    separate_trains = [False]
    groups = [5]
    train_max_episode = 200
    train_max_steps = train_max_episode * 100
    
    # 其余参数（如果这个脚本仅用于触发 evaluate，训练参数其实用不上，写着防报错即可）
    task = {'type': 'top',
            'evaluate_optimizers': evaluate_optimizers,
            'base_evaluate_optimizers': base_evaluate_optimizers,
            'evaluate_function': evaluate_function,  # 报错就是因为原来缺了这一行
            'runtimes': runtimes,
            'separate_trains': separate_trains,
            'groups': groups,
            'train_max_episode': train_max_episode,
            'train_max_steps': train_max_steps,
            'dims': dims,
            'train_times': 1,
            'lr_critic': 1e-4,
            'lr_actor': 1e-6,
            }

    return [task]


if __name__ == '__main__':
    print(all_tasks_generate())