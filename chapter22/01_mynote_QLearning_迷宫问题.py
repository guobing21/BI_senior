"""
模拟迷宫问题
"""
import numpy as np

#贴现因子，是个超参数
GAMMA = 0.8
# 初始化的Q函数（动作价值函数）
Q = np.zeros((6, 6))
# 环境给的奖励reward
# -1代表两个房间不连通，0代表联通，100代表出了迷宫
R = np.asarray(
    [[-1, -1, -1, -1, 0, -1],
     [-1, -1, -1, 0, -1, 100],
     [-1, -1, -1, 0, -1, -1],
     [-1, 0, 0, -1, 0, -1],
     [0, -1, -1, 0, -1, 100],
     [-1, 0, -1, -1, 0, 100]])

# [[  0.    0.    0.    0.   80.    0. ]
#  [  0.    0.    0.   64.    0.  100. ]
#  [  0.    0.    0.   64.    0.    0. ]
#  [  0.   80.   51.2   0.   80.    0. ]
#  [ 64.    0.    0.   64.    0.  100. ]
#  [  0.   80.    0.    0.   80.  100. ]]

# 取每一行的最大值(当前状态的最大收益)
def getMaxQ(state):
    #print(state)
    # 通过选取最大动作值来进行最优策略学习
    return max(Q[state, :])

# QLearning函数
def QLearning(state):
    #选择的动作
    curAction = None
    # 遍历所有的结点，查看是否可以移动到下一步
    for action in range(6):
        #如果不能走，价值函数为0
        if (R[state][action] == -1):
            Q[state, action] = 0
        # 如果可以走，记录curAction,
        # 并且等于curAction最大的贴现+当前的反馈
        else:
            curAction = action
            # 选择动作最大的
            # 环境给的当下的奖励 + 未来最大收益的贴现
            Q[state, action] = R[state, action] + GAMMA*getMaxQ(curAction)

# 模拟1000次
for count in range(1000):
    # 一个episode：遍历所有的结点(0-5)
    for i in range(6):
        # 更新结点i的QLearning
        QLearning(i)

# 显示保留小数点后一位
np.set_printoptions(precision=1)
print(Q/5)
# Q表就是动作价值表
# [[  0.    0.    0.    0.   80.    0. ]
#  [  0.    0.    0.   64.    0.  100. ]
#  [  0.    0.    0.   64.    0.    0. ]
#  [  0.   80.   51.2   0.   80.    0. ]
#  [ 64.    0.    0.   64.    0.  100. ]
#  [  0.   80.    0.    0.   80.  100. ]]