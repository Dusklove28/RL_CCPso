from matAgent.ccpso_50d import FiftyDimCCPsoSwarm
from matAgent.testpso import TestpsoSwarm


def all_tasks_generate():
    evaluate_optimizers = [
        TestpsoSwarm,
        FiftyDimCCPsoSwarm,
    ]

    task = {
        'type': 'top',
        'evaluate_optimizers': evaluate_optimizers,
        'evaluate_function': [1, 5, 10, 20],
        'runtimes': 5,
        'separate_trains': [False],
        'groups': [5],
        'train_max_episode': 200,
        'train_max_steps': 200 * 100,
        'dims': [50],
        'train_times': 1,
        'max_fe': int(1e4),
        'n_part': 100,
        'lr_critic': 1e-7,
        'lr_actor': 1e-9,
    }

    return [task]


if __name__ == '__main__':
    print(all_tasks_generate())
