import numpy as np
from matAgent.ccpso_50d import FiftyDimCCPsoSwarm
from rl.DDPG.TF2_DDPG_Basic import DDPG
from env.TestpsoEnv import TestpsoEnv


class RlCCPsoSwarm(FiftyDimCCPsoSwarm):
    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'RL-CCPso50D'

        # 从配置字典中获取模型路径
        model_path = config_dic.get('model')
        if not model_path:
            raise ValueError("评估 RL-CCPso50D 必须提供 model 路径！")

        # 初始化一个用于构建网络维度的影子环境
        gym_env = TestpsoEnv(show=False, al_type='ccpso_50d')

        # 实例化 DDPG 并加载你训练好的参数 (注意：网络结构必须和训练时完全一致)
        ddpg = DDPG(gym_env, discrete=False, memory_cap=1000, actor_units=(16,),
                    critic_units=(8, 16, 32))
        ddpg.load_actor(model_path)
        self.ddpg_actor = ddpg

    def run_once(self, action=None):
        # 评估阶段，算法自己获取状态，自己去网络里查询动作，然后执行
        state = self.get_state()
        action = self.ddpg_actor.policy(state)
        super().run_once(action.numpy())