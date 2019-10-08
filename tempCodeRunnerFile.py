tmp = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))]).T    # 求x1和x2的笛卡尔积
# x1 = tmp[0]
# x2 = tmp[1]