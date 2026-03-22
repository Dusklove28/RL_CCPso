from log import logger
from rl.DDPG.TF2_DDPG_Basic import DDPG


ORIGINAL_RLEPSO_ACTOR_UNITS = (16, 32, 32, 32, 64, 64)
ORIGINAL_RLEPSO_CRITIC_UNITS = (8, 16, 32, 32, 16, 8)
ORIGINAL_RLEPSO_LR_CRITIC = 1e-7
ORIGINAL_RLEPSO_LR_ACTOR = 1e-9


def build_original_rlepso_ddpg(
        env,
        discrete=False,
        noise='norm',
        sigma=0.15,
        tau=0.125,
        gamma=0.85,
        batch_size=64,
        memory_cap=100000):
    return DDPG(
        env,
        discrete=discrete,
        memory_cap=memory_cap,
        actor_units=ORIGINAL_RLEPSO_ACTOR_UNITS,
        critic_units=ORIGINAL_RLEPSO_CRITIC_UNITS,
        use_priority=True,
        lr_critic=ORIGINAL_RLEPSO_LR_CRITIC,
        lr_actor=ORIGINAL_RLEPSO_LR_ACTOR,
        noise=noise,
        sigma=sigma,
        tau=tau,
        gamma=gamma,
        batch_size=batch_size,
    )


def get_ddpg_object(
        env,
        discrete=False,
        use_priority=True,
        lr_actor=1e-5,
        lr_critic=1e-3,
        actor_units=None,
        critic_units=None,
        noise='norm',
        sigma=0.15,
        tau=0.125,
        gamma=0.85,
        batch_size=64,
        memory_cap=100000):
    # 保持 RL_testpso 原训练条件不变，其他算法适配这套设置。
    return build_original_rlepso_ddpg(
        env,
        discrete=discrete,
        noise=noise,
        sigma=sigma,
        tau=tau,
        gamma=gamma,
        batch_size=batch_size,
        memory_cap=memory_cap,
    )


def train():
    """
    仅供本地调试使用；task 框架不会调用这里。
    """
    from env.TestpsoEnv import TestpsoEnv

    algo_type = 'ccpso_50d'
    logger.info(f"==== 本地调试训练，当前挂载算法: {algo_type} ====")

    gym_env = TestpsoEnv(show=False, al_type=algo_type)

    try:
        assert gym_env.action_space.high == -gym_env.action_space.low
        is_discrete = False
        logger.info('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        logger.info('Discrete Action Space')

    ddpg = get_ddpg_object(
        gym_env,
        discrete=is_discrete,
        memory_cap=10000000,
        noise='norm',
        sigma=0.15,
        tau=0.125,
        gamma=0.85,
    )

    max_episodes = 200
    max_steps = 1000
    logger.info(f"开启底层训练循环，总回合数: {max_episodes} ...")
    ddpg.train(max_episodes=max_episodes, max_steps=max_steps)


def test():
    pass


if __name__ == "__main__":
    logger.info('独立运行 ddpg.py 测试启动')
    train()
