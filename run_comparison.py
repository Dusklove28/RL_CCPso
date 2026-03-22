import numpy as np
from env.TestpsoEnv import TestpsoEnv
from rl.DDPG.TF2_DDPG_Basic import DDPG
from matAgent.pso import PsoSwarm
from matAgent.clpso import ClpsoSwarm
from functions import CEC_functions

# --- 配置区 ---
DIM = 50  # 维数
N_PART = 40  # 粒子数
MAX_FES = 20000  # 最大评价次数
RUN_TIMES = 30  # 每个函数独立跑 30 次取平均
FUNCTIONS_TO_TEST = range(1, 29)  # 测试 CEC 1 到 28

# 你的两个训练好的 h5 模型路径 (请根据实际情况修改)
MODEL_TESTPSO = "ddpg_actor_testpso.h5"
MODEL_CCPSO = "ddpg_actor_ccpso50d.h5"


def run_rl_optimizer(al_type, model_path, f_num):
    """运行强化学习控制的优化器"""
    # 初始化环境，传入指定的固定函数
    env = TestpsoEnv(show=False, al_type=al_type, fixed_fun_num=f_num)

    # 初始化 DDPG 并加载训练好的权重 (参数要和训练时一致)
    is_discrete = False
    ddpg = DDPG(env, discrete=is_discrete, memory_cap=1000000, actor_units=(16,),
                critic_units=(8, 16, 32))
    ddpg.load_actor(model_path)

    # 跑一个完整的 Episode
    state = env.reset()
    done = False
    while not done:
        # RL 网络根据当前 state 输出 action
        action = ddpg.policy(state)
        state, reward, done, _ = env.step(action)

    return env.pso_swarm.history_best_fit


def run_classic_optimizer(OptimizerClass, f_num):
    """运行经典的基准 PSO"""
    cec_functions = CEC_functions(DIM)

    # 包装测试函数
    def test_fun(x):
        if len(x.shape) == 2:
            ans = [cec_functions.Y(row, f_num) for row in x]
        else:
            ans = cec_functions.Y(x, f_num)
        return np.array(ans)

    # 初始化经典 PSO 并运行
    # 注意：经典 PSO 的 n_run 通常是 max_fes / n_part
    n_run = int(MAX_FES / N_PART)
    optimizer = OptimizerClass(n_run=n_run, n_part=N_PART, show=False,
                               fun=test_fun, n_dim=DIM, pos_max=100, pos_min=-100,
                               config_dic={'max_fes': MAX_FES})
    optimizer.run()

    return optimizer.history_best_fit


if __name__ == '__main__':
    print("=== 开始严格对比实验 ===")

    for f_num in FUNCTIONS_TO_TEST:
        print(f"\n--- 正在测试 CEC 函数 F{f_num} ---")

        results = {
            'RL-TestPso': [],
            'RL-CCPso50D': [],
            'Classic-PSO': [],
            'Classic-CLPSO': []
        }

        for run in range(RUN_TIMES):
            # 1. 测试你的 CCPso
            best_ccpso = run_rl_optimizer('ccpso_50d', MODEL_CCPSO, f_num)
            results['RL-CCPso50D'].append(best_ccpso)

            # 2. 测试原版的 RL PSO
            best_testpso = run_rl_optimizer('testpso', MODEL_TESTPSO, f_num)
            results['RL-TestPso'].append(best_testpso)

            # 3. 测试经典基础 PSO
            best_pso = run_classic_optimizer(PsoSwarm, f_num)
            results['Classic-PSO'].append(best_pso)

            # 4. 测试经典 CLPSO
            best_clpso = run_classic_optimizer(ClpsoSwarm, f_num)
            results['Classic-CLPSO'].append(best_clpso)

            print(f"  第 {run + 1}/{RUN_TIMES} 次跑完...", end='\r')

        print("\n[最终成绩 - 30次独立运行均值 (越小越好)]:")
        print(f"  RL-CCPso50D  : {np.mean(results['RL-CCPso50D']):.4e} ± {np.std(results['RL-CCPso50D']):.2e}")
        print(f"  RL-TestPso   : {np.mean(results['RL-TestPso']):.4e} ± {np.std(results['RL-TestPso']):.2e}")
        print(f"  Classic-PSO  : {np.mean(results['Classic-PSO']):.4e} ± {np.std(results['Classic-PSO']):.2e}")
        print(f"  Classic-CLPSO: {np.mean(results['Classic-CLPSO']):.4e} ± {np.std(results['Classic-CLPSO']):.2e}")