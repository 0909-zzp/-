import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf
from collections import Counter
import networkx as nx
import pyreadr


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False

# ==================== 路径设置 ====================
DATA_DIR = r"C:\Users\27438\Desktop\大创\图形Lasso"
OUTPUT_DIR = os.path.join(DATA_DIR, "figures_filtered")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 读取数据 ====================
weights_file = os.path.join(DATA_DIR, "reg_weights_2436.csv")
cond_file = os.path.join(DATA_DIR, "reg_cond_2436.csv")

reg_weights = pd.read_csv(weights_file, header=None, skiprows=1).values.astype(np.float32)
reg_cond = pd.read_csv(cond_file)['cond'].values

# 异常交易日识别与剔除
daily_max = reg_weights.max(axis=1)
abnormal_cond = reg_cond > 500
abnormal_weight = daily_max > 0.1
abnormal_days = np.where(abnormal_cond | abnormal_weight)[0]
print(f"异常交易日数量: {len(abnormal_days)}")
print(f"异常交易日索引: {abnormal_days[:10]}...")

mask = ~(abnormal_cond | abnormal_weight)
filtered_weights = reg_weights[mask]
filtered_cond = reg_cond[mask]
print(f"剔除后剩余交易日: {len(filtered_weights)}")
print(f"剔除后权重最大值: {filtered_weights.max():.4f}")
print(f"剔除后条件数均值: {filtered_cond.mean():.2f}")

T, K = filtered_weights.shape
print(f"最终样本交易日数: {T}, 资产数: {K}")

# ==================== 1. 整体权重统计 ====================
all_w = filtered_weights.flatten()
print("\n=== 整体权重统计 ===")
print(f"最小值: {all_w.min():.6f}")
print(f"最大值: {all_w.max():.6f}")
print(f"均值: {all_w.mean():.6f}")
print(f"标准差: {all_w.std():.6f}")
print(f"偏度: {skew(all_w):.4f}")
print(f"峰度: {kurtosis(all_w):.4f}")

# 直方图（密度）
plt.figure(figsize=(8,5))
plt.hist(all_w, bins=100, density=False, alpha=0.7, edgecolor='black')
plt.title('正则化GMVP权重分布（剔除异常日后）')
plt.xlabel('权重')
plt.ylabel('频数')   # 修正：频数
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'weight_histogram.png'), dpi=150)
plt.close()

# ==================== 2. 每日权重和验证 ====================
row_sums = filtered_weights.sum(axis=1)
print("\n=== 每日权重和 ===")
print(f"最小和: {row_sums.min():.10f}")
print(f"最大和: {row_sums.max():.10f}")
print(f"均值: {row_sums.mean():.10f}")
print(f"偏差最大值: {np.abs(row_sums - 1).max():.2e}")

# ==================== 3. 个股统计 ====================
stock_means = filtered_weights.mean(axis=0)
stock_stds = filtered_weights.std(axis=0)
print("\n=== 个股统计 ===")
print(f"个股均值范围: [{stock_means.min():.6f}, {stock_means.max():.6f}]")
print(f"个股标准差范围: [{stock_stds.min():.6f}, {stock_stds.max():.6f}]")
print(f"所有个股平均波动率: {stock_stds.mean():.6f}")

# 计算每日权重的最大值（反映集中度）
daily_max_weight = filtered_weights.max(axis=1)  # 每日最大权重
# 计算每日权重的最小值
daily_min_weight = filtered_weights.min(axis=1)  # 每日最小权重
# 计算每日权重的标准差（反映分散程度）
daily_std = filtered_weights.std(axis=1)         # 每日权重标准差
# 计算每日权重的熵（反映分布均匀性）
def calculate_entropy(weights_row):
    # 只考虑正值权重，避免log(0)
    pos_weights = weights_row[weights_row > 0]
    if len(pos_weights) == 0:
        return 0
    pos_weights = pos_weights / pos_weights.sum()  # 重新归一化
    entropy = -np.sum(pos_weights * np.log(pos_weights + 1e-10))
    return entropy

daily_entropy = np.array([calculate_entropy(row) for row in filtered_weights])

print("\n=== 时间序列特征 ===")
print(f"每日最大权重均值: {daily_max_weight.mean():.6f}")
print(f"每日最大权重标准差: {daily_max_weight.std():.6f}")
print(f"每日权重标准差均值: {daily_std.mean():.6f}")
print(f"每日最大权重范围: {daily_max_weight.min():.6f} ~ {daily_max_weight.max():.6f}")
print(f"每日最小权重范围: {daily_min_weight.min():.6f} ~ {daily_min_weight.max():.6f}")
print(f"每日权重标准差范围: {daily_std.min():.6f} ~ {daily_std.max():.6f}")
print(f"每日熵值均值: {daily_entropy.mean():.6f}")


# 绘制时间序列特征（一个大图包含5个子图，2x3布局，第6个位置留空或可加其他）
plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
plt.plot(daily_max_weight, alpha=0.7)
plt.title('每日最大权重')
plt.xlabel('交易日')
plt.ylabel('最大权重')
plt.grid(alpha=0.3)

plt.subplot(2,3,2)
plt.plot(daily_std, alpha=0.7, color='orange')
plt.title('每日权重标准差（分散度）')
plt.xlabel('交易日')
plt.ylabel('标准差')
plt.grid(alpha=0.3)

plt.subplot(2,3,3)
plt.plot(daily_entropy, alpha=0.7, color='green')
plt.title('每日权重分布熵')
plt.xlabel('交易日')
plt.ylabel('熵值')
plt.grid(alpha=0.3)

# 绘制每日权重范围
plt.subplot(2,3,4)
plt.plot(daily_min_weight, alpha=0.7, color='red', label='最小权重')
plt.plot(daily_max_weight, alpha=0.7, color='blue', label='最大权重')
plt.title('每日权重范围')
plt.xlabel('交易日')
plt.ylabel('权重')
plt.legend()
plt.grid(alpha=0.3)

# 绘制每日权重分布的散点图（抽样）
sample_indices = np.linspace(0, len(filtered_weights)-1, min(50, len(filtered_weights)), dtype=int)
sample_weights = filtered_weights[sample_indices].flatten()
sample_dates = np.repeat(range(len(sample_indices)), K)

plt.subplot(2,3,5)
plt.scatter(sample_dates, sample_weights, alpha=0.3, s=1)
plt.title('权重时间序列散点图（抽样）')
plt.xlabel('交易日')
plt.ylabel('权重')
plt.grid(alpha=0.3)

# 第6个子图留白
plt.subplot(2,3,6)
plt.axis('off')

plt.tight_layout()

# 保存
plt.savefig(os.path.join(OUTPUT_DIR, 'time_series_extended.png'), dpi=150)
plt.close()


# ==================== 5. 自相关函数包络图 ====================
max_lag = 300
acf_all = np.zeros((K, max_lag))
for i in range(K):
    acf_vals = acf(filtered_weights[:, i], nlags=max_lag, fft=False)
    acf_all[i, :] = acf_vals[1:max_lag+1]
quantiles = np.percentile(acf_all, [5, 50, 95], axis=0)
lower, median, upper = quantiles[0], quantiles[1], quantiles[2]
conf = 1.96 / np.sqrt(T)
lags = np.arange(1, max_lag+1)

plt.figure(figsize=(12,6))
plt.fill_between(lags, lower, upper, color='gray', alpha=0.3, label='5%-95% 分位数范围')
plt.plot(lags, median, 'b-', linewidth=1.5, label='中位数')
plt.axhline(conf, color='r', linestyle='--', linewidth=1, label='Bartlett 置信区间')
plt.axhline(-conf, color='r', linestyle='--', linewidth=1)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('滞后阶数')
plt.ylabel('自相关系数')
plt.title('正则化后 GMVP 权重的自相关函数包络图（剔除异常日后）')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'acf_envelope.png'), dpi=150)
plt.close()
print(f"\n滞后1阶中位数自相关系数: {median[0]:.4f}")

# ==================== 6. 条件数分析 ====================
print(f"\n条件数均值: {filtered_cond.mean():.2f}")
print(f"最小值: {filtered_cond.min():.2f}, 最大值: {filtered_cond.max():.2f}")
print(f"中位数: {np.median(filtered_cond):.2f}")


# # ==================== 原始条件数（剔除后样本）与正则化后对比（两个子图） ====================
# print("\n正在计算剔除后样本的原始协方差矩阵条件数（串行，避免多进程错误）...")
# DATA_DIR_MIN = r"C:\Users\27438\Desktop\大创\1min"
# files = [os.path.join(DATA_DIR_MIN, f) for f in os.listdir(DATA_DIR_MIN) if f.endswith('_1min_log_return.RData')]
# files.sort()
#
# # 获取剔除后样本的原始索引
# normal_indices = np.where(mask)[0]
#
# orig_cond_filtered = []
# total = len(normal_indices)
# for i, idx in enumerate(normal_indices):
#     if i % 200 == 0:
#         print(f"  进度: {i}/{total}")
#     result = pyreadr.read_r(files[idx])
#     df = list(result.values())[0]
#     returns = df.T.values
#     cov = returns.T @ returns
#     orig_cond_filtered.append(np.linalg.cond(cov))
# orig_cond_filtered = np.array(orig_cond_filtered)
#
# print(f"原始条件数（剔除后样本）均值: {np.mean(orig_cond_filtered):.2e}")
# print(f"原始条件数（剔除后样本）中位数: {np.median(orig_cond_filtered):.2e}")
#
# # 核密度估计
# from scipy.stats import gaussian_kde
# log_orig = np.log10(orig_cond_filtered)
# log_reg = np.log10(filtered_cond)
#
# kde_orig = gaussian_kde(log_orig)
# kde_reg = gaussian_kde(log_reg)
#
# # 定义绘图范围（各自使用数据的 min/max，也可统一范围）
# x_orig = np.linspace(log_orig.min(), log_orig.max(), 200)
# x_reg = np.linspace(log_reg.min(), log_reg.max(), 200)
#
# # 创建两个子图（左右并排）
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#
# # 左图：原始条件数
# ax1.plot(x_orig, kde_orig(x_orig), 'r-', linewidth=2)
# ax1.fill_between(x_orig, 0, kde_orig(x_orig), alpha=0.3, color='red')
# ax1.set_xlabel('log10(条件数)')
# ax1.set_ylabel('密度')
# ax1.set_title('原始协方差矩阵条件数')
# ax1.grid(alpha=0.3, linestyle='--')
#
# # 右图：正则化后条件数
# ax2.plot(x_reg, kde_reg(x_reg), 'b-', linewidth=2)
# ax2.fill_between(x_reg, 0, kde_reg(x_reg), alpha=0.3, color='blue')
# ax2.set_xlabel('log10(条件数)')
# ax2.set_ylabel('密度')
# ax2.set_title('图形Lasso后条件数')
# ax2.grid(alpha=0.3, linestyle='--')
#
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'cond_kde_comparison_two_panels.png'), dpi=150)
# plt.close()
# print("核密度对比图（两个子图）已保存至 cond_kde_comparison_two_panels.png")


# ==================== 7. 资产网络分析 ====================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx

DATA_DIR = r"C:\Users\27438\Desktop\大创\图形Lasso"
OUTPUT_DIR = os.path.join(DATA_DIR, "figures_filtered")
os.makedirs(OUTPUT_DIR, exist_ok=True)

prec_file = os.path.join(DATA_DIR, "prec_last_normal.csv")
prec_last = pd.read_csv(prec_file, header=None, skiprows=1).values.astype(np.float32)
K = prec_last.shape[0]
print(f"精度矩阵维度: {K}×{K}")

# 构建邻接矩阵（阈值 0.1）
threshold = 0.1
adj = np.zeros((K, K))
for i in range(K):
    for j in range(i+1, K):
        pcorr = -prec_last[i, j] / np.sqrt(prec_last[i, i] * prec_last[j, j])
        if abs(pcorr) > threshold:
            adj[i, j] = adj[j, i] = 1

n_edges = int(adj.sum() / 2)
density = n_edges / (K * (K - 1) / 2)
degrees = adj.sum(axis=1)
print(f"边数: {n_edges}, 密度: {density:.6f}")
print(f"度分布: 最小值 {degrees.min()}, 最大值 {degrees.max()}, 均值 {degrees.mean():.2f}")

# 度分布直方图
plt.figure(figsize=(8,5))
plt.hist(degrees, bins=30, edgecolor='black')
plt.title('节点度分布（最后一个正常日）')
plt.xlabel('度')
plt.ylabel('频数')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'degree_distribution_normal.png'), dpi=150)
plt.close()

# 双对数度分布
pos_degrees = degrees[degrees > 0]
if len(pos_degrees) > 1:
    cnt = Counter(pos_degrees)
    x = sorted(cnt.keys())
    y = [cnt[d] for d in x]
    plt.figure(figsize=(6,4))
    plt.loglog(x, y, 'bo', markersize=6, label='观测度分布')
    if len(x) > 2:
        try:
            logx = np.log(x)
            logy = np.log(y)
            coeffs = np.polyfit(logx, logy, 1)
            fitted = np.exp(coeffs[1]) * np.array(x) ** coeffs[0]
            plt.loglog(x, fitted, 'r-', label=f'幂律拟合 (γ={-coeffs[0]:.2f})')
        except Exception as e:
            print("幂律拟合失败:", e)
    plt.xlabel('度')
    plt.ylabel('频数')
    plt.title('度分布（双对数坐标）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'degree_distribution_loglog_normal.png'), dpi=150)
    plt.close()
else:
    print("无大于0的度，无法绘制双对数图")

# 中心性指标
G = nx.from_numpy_array(adj)
degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
closeness_centrality = nx.closeness_centrality(G)
centrality_df = pd.DataFrame({
    'degree': list(degree_centrality.values()),
    'eigenvector': list(eigenvector_centrality.values()),
    'betweenness': list(betweenness_centrality.values()),
    'closeness': list(closeness_centrality.values())
})
centrality_df.to_csv(os.path.join(DATA_DIR, 'network_centrality_normal.csv'), index=False)

print("网络分析完成，图表已保存。")



print(f"\n所有图表已保存至 {OUTPUT_DIR}")
print("描述性分析和网络分析完成。")



#============================权重分布直方对比图==============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyreadr

# ==================== 路径设置 ====================
DATA_DIR_RAW = r"C:\Users\27438\Desktop\大创\1min"           # 原始分钟数据
DATA_DIR_RES = r"C:\Users\27438\Desktop\大创\图形Lasso"      # 图形Lasso结果
OUTPUT_DIR = os.path.join(DATA_DIR_RES, "figures_filtered")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 计算原始 GMVP 权重（前 500 天） ====================
files = [os.path.join(DATA_DIR_RAW, f) for f in os.listdir(DATA_DIR_RAW) if f.endswith('_1min_log_return.RData')]
files.sort()
N_DAYS = 500   # 取前 500 天，避免计算过慢
files_sample = files[:N_DAYS]

print("正在计算原始 GMVP 权重（前 {} 天）...".format(N_DAYS))
orig_weights_list = []
for i, f in enumerate(files_sample):
    if i % 100 == 0:
        print(f"  进度: {i+1}/{N_DAYS}")
    result = pyreadr.read_r(f)
    df = list(result.values())[0]
    returns = df.T.values          # (390, 392)
    cov = returns.T @ returns
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(cov.shape[0])
    w = inv_cov @ ones
    w = w / w.sum()
    orig_weights_list.append(w)
orig_weights = np.array(orig_weights_list)   # (N_DAYS, 392)
orig_weights_flat = orig_weights.flatten()

print("\n原始权重统计（前 {} 天）:".format(N_DAYS))
print(f"  最小值: {orig_weights_flat.min():.2f}")
print(f"  最大值: {orig_weights_flat.max():.2f}")
print(f"  均值: {orig_weights_flat.mean():.2f}")
print(f"  标准差: {orig_weights_flat.std():.2f}")

# ==================== 读取正则化权重（全量，取前 N_DAYS 天） ====================
reg_weights_full = pd.read_csv(os.path.join(DATA_DIR_RES, "reg_weights_2436.csv"), header=None, skiprows=1).values.astype(np.float32)
reg_weights_sample = reg_weights_full[:N_DAYS]   # 前 N_DAYS 天
reg_flat = reg_weights_sample.flatten()

print("\n正则化权重统计（前 {} 天）:".format(N_DAYS))
print(f"  最小值: {reg_flat.min():.6f}")
print(f"  最大值: {reg_flat.max():.6f}")
print(f"  均值: {reg_flat.mean():.6f}")
print(f"  标准差: {reg_flat.std():.6f}")

# ==================== 绘制对比直方图 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：原始权重（对数 y 轴）
ax1.hist(orig_weights_flat, bins=100, alpha=0.7, color='red', edgecolor='black')
ax1.set_yscale('log')
ax1.set_xlabel('权重')
ax1.set_ylabel('频数 (log scale)')
ax1.set_title(f'原始 GMVP 权重分布（前 {N_DAYS} 天）')
ax1.grid(alpha=0.3)

# 右图：正则化权重（线性 y 轴）
ax2.hist(reg_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
ax2.set_xlabel('权重')
ax2.set_ylabel('频数')
ax2.set_title(f'正则化 GMVP 权重分布（前 {N_DAYS} 天）')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'orig_vs_reg_weights_hist.png'), dpi=150)
plt.show()

print("对比图已保存至", os.path.join(OUTPUT_DIR, 'orig_vs_reg_weights_hist.png'))
