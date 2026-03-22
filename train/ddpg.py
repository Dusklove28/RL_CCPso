from log import logger  # 引入我们自定义的日志器
from rl.DDPG.TF2_DDPG_Basic import DDPG
import numpy as np


def get_ddpg_object(
        env,
        discrete=False,
        use_priority=False,
        lr_actor=1e-5,
        lr_critic=1e-3,
        actor_units=(24, 16),
        critic_units=(24, 16),
        noise='norm',
        sigma=0.15,
        tau=0.125,
        gamma=0.85,
        batch_size=64,
        memory_cap=100000):
    # 【核心】：框架真正调用的是这里。
    # 保持极小的学习率 1e-7 和 1e-9，保证对比实验的公平性。
    return DDPG(env, discrete=discrete, memory_cap=memory_cap, actor_units=(16, 32, 32, 32, 64, 64),
                critic_units=(8, 16, 32, 32, 16, 8), use_priority=True, lr_critic=1e-7, lr_actor=1e-9)


def train():
    """
    注意：这个函数只是供你如果不走 task 框架时，本地临时跑着玩、查 Bug 用的。
    自动化框架并不会执行这里。但为了代码严谨，我们把它改对。
    """
    from env.TestpsoEnv import TestpsoEnv

    ALGO_TYPE = 'ccpso_50d'
    logger.info(f"==== 本地调试训练，当前挂载算法: {ALGO_TYPE} ====")

    gym_env = TestpsoEnv(show=False, al_type=ALGO_TYPE)

    try:
        assert (gym_env.action_space.high == -gym_env.action_space.low)
        is_discrete = False
        logger.info('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        logger.info('Discrete Action Space')

    # 保持原版配置
    ddpg = DDPG(gym_env, discrete=is_discrete, memory_cap=10000000, actor_units=(16,),
                critic_units=(8, 16, 32), use_priority=True, lr_critic=1e-7, lr_actor=1e-9)

    max_episodes = 200
    max_steps = 1000
    logger.info(f"开启底层训练循环，总回合数: {max_episodes} ...")
    ddpg.train(max_episodes=max_episodes, max_steps=max_steps)


def test():
    """
    既然我们全面使用 task 自动化评估框架 (evluate_optimizer.py)，
    这个本地简陋的测试函数直接 pass 废弃即可，免得产生误导。
    """
    pass


if __name__ == "__main__":
    logger.info('独立运行 ddpg.py 测试启动')
    train()