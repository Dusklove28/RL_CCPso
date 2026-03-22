import numpy as np

from matAgent.ccpso_50d import FiftyDimCCPsoSwarm


class RlCCPsoSwarm(FiftyDimCCPsoSwarm):
    optimizer_name = 'RL_CCPSO50D'

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        config_dic = {} if config_dic is None else dict(config_dic)
        model_path = config_dic.get('model')
        if not model_path:
            raise ValueError("评估 RL-CCPso50D 必须提供 model 路径")

        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'RL-CCPso50D'
        self.optimizer_name = self.name

    def run_once(self, action=None):
        state = self.get_state()
        action = self.ddpg_actor.policy(state)
        super().run_once(action.numpy())
