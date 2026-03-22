import numpy as np
from matAgent.testpso import TestpsoSwarm  # 直接继承原版，复用它的 CLPSO/FDR 目标计算逻辑


class FiftyDimCCPsoSwarm(TestpsoSwarm):
    optimizer_name = 'CCPSO_50D'
    action_space = 10
    obs_space = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = '50D_CC_PSO'
        # 统一式特需：记录上一代的位置 X(t-1)
        self.xs_old = self.xs.copy()

    def run_once(self, actions):
        if actions is None:
            actions = np.zeros(self.action_space * self.n_group)
        if self.show:
            print('{}|best fit:{}'.format(self.fe_num / self.fe_max, self.history_best_fit))

        new_xs = np.zeros_like(self.xs)

        for i in range(self.n_part):
            fdr_deta_fitness = self.atom_history_best_fits[i] - self.atom_history_best_fits

            # 和训练时的 group 配置保持一致
            action = actions[
                i % self.n_group * self.action_space:
                i % self.n_group * self.action_space + self.action_space
            ]

            w = action[7] * 0.4 + 0.5
            r1 = action[1] * 1.5 + 1.5
            r2 = action[2] * 1.5 + 1.5
            r5 = action[5] * 1.5 + 1.5
            r6 = action[6] * 1.5 + 1.5

            # 【核心改动 1】：提取 action[0] 作为收敛性控制目标 P_ECon
            P_ECon = action[0] * 0.5 + 0.5  # 映射到 [0, 1] 之间

            r = np.array([r1, r2, 0, 0, r5, r6])
            r = r / (np.sum(r) + 1e-10) * (action[8] + 1) * 4
            r1, r2, r3, r4, r5, r6 = r

            mutation_rate = (action[9] + 1) * 0.01

            for d in range(self.n_dim):
                # 完全保留原版的异构目标寻找逻辑
                clpso_target = self.p_best[self.fid[i, d], d]

                xid = self.xs[i, d]
                distance = xid - self.p_best[:, d]
                distance[i] = np.inf
                fdr = fdr_deta_fitness / (distance + 1e-250)
                j_index = np.argmax(fdr)
                fdr_target = self.p_best[j_index, d]

                gbest_target = self.g_best[d]
                pbest_target = self.p_best[i, d]

                # 【核心改动 2】：计算总引力 C 和等效中心 Q
                C = r1 + r2 + r5 + r6

                # 计算目标点的加权中心
                Q = (r1 * clpso_target + r2 * fdr_target + r5 * gbest_target + r6 * pbest_target) / (C + 1e-16)

                # 计算统一式系数
                a1 = 1 + w - C
                a2 = -w

                # 计算自然期望偏差 X_Q
                X_Q = a1 * (self.xs[i, d] - Q) + a2 * (self.xs_old[i, d] - Q)

                # 计算一维的实际收敛度 P_Con (避免除零)
                P_Con = np.abs(X_Q)
                if P_Con == 0:
                    P_Con = 1e-16

                # 应用收敛性控制公式更新当前维度的位置
                new_xs[i, d] = Q + (P_ECon / P_Con) * X_Q

            # 保留原版的突变逻辑以求公平
            if np.random.random() < mutation_rate * self.flag[i]:
                new_xs[i] = np.random.uniform(self.pos_min, self.pos_max, self.xs[i].shape)

        # 越界处理
        new_xs = np.clip(new_xs, self.pos_min, self.pos_max)

        # 状态迭代
        self.xs_old = self.xs.copy()
        self.xs = new_xs

        self.fits = self.fun(self.xs)
        self.update_best()
