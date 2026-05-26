import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime

import binance_fetcher

plt.rcParams['font.sans-serif'] = ['SimHei','DejaVuSans','Microsoft YaHei','Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False#解决负号问题
plt.rcParams['mathtext.fontset'] = 'stix'#stix数学字体


df=pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\vol-surface-engine\data\data.csv')
df=df[df['symbol'].str.startswith('BTC')].copy()
print(f"{len(df)}：筛选出来的BTC行数")

#-------------------计算SVI需要的变量--------------------------
#svi公式：w(k) = a + b * ( rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) )
df['T'] = df['T'].astype(float)
df['w'] = df['implied_vol'] ** 2 * df['T']
#查看到期日的k范围
print("\n各到期日的k范围")
for expiry in df['expiry_date'].unique():
    subset=df[df['expiry_date'] == expiry]
    print(
        f"  {expiry}: k ∈ [{subset['log_moneyness'].min():.3f}, {subset['log_moneyness'].max():.3f}], 合约数: {len(subset)}")
    #分组聚合
grouped=df.groupby(['expiry_date','strike']).agg({
    'log_moneyness': 'first',  # k 是一样的
    'w': 'mean',  # 取平均
    'T': 'first',  # T 是一样的
    'implied_vol': 'mean'  # 保留用于参考
}).reset_index()#agg()之后会让expiry_date和strike变成索引，为了能正常用df方法访问列值，所以需要reset_index
print(f"\n合并后：{len(grouped)}行")
expiry_example=df['expiry_date'].unique()[0]
sample=grouped[grouped['expiry_date'] == expiry_example].sort_values('strike')
print(sample.head(10))
#------------------------定义svi函数--------------------------
def svi_w(k,a,b,rho,m,sigma):
    sqrt_term=np.sqrt((k-m)**2+sigma**2)
    return a+b*(rho*(k-m)+sqrt_term)
def svi_iv(k,a,b,rho,m,sigma,T):
    w=svi_w(k,a,b,rho,m,sigma)
    w=np.maximum(w,0)
    return np.sqrt(w/T)
test_k=np.array([-0.5,0,0.5])
test_w = svi_w(test_k, a=0.01, b=0.3, rho=-0.3, m=0, sigma=0.2)
print("测试 SVI 函数:")
for k, w in zip(test_k, test_w):
    print(f"  k={k:.2f} -> w={w:.5f}")
#-----------定义损失函数和权重-----------------
def svi_loss(params, k_obs, w_obs, weights=None):#后续可以考虑加上正则化
    """SVI 拟合的损失函数"""
    a, b, rho, m, sigma = params

    # 软约束惩罚
    penalty = 0
    if b < 0:
        penalty += 100 * (-b) ** 2
    if abs(rho) >= 0.999:
        penalty += 100 * (abs(rho) - 0.999) ** 2
    if sigma <= 0:
        penalty += 100 * (-sigma + 1e-6) ** 2

    w_pred = svi_w(k_obs, a, b, rho, m, sigma)

    if weights is None:
        loss = np.mean((w_obs - w_pred) ** 2)
    else:
        loss = np.mean(weights * (w_obs - w_pred) ** 2)
    loss=loss*1e6

    return loss + penalty
def get_weights(k_obs):#这里的k_obs需要传入列表或者是数组
    sigma_k=0.1
    weights=np.exp(-k_obs**2/(2*sigma_k**2))#这是一个高斯核，是一个描述距离越近，相似度越高的权重函数
    return weights/weights.sum()#.sum()方法只有numpy.array和pd.dataframe可以用
#------------------参数初始化------------------------
def initialize_svi_params(k_obs,w_obs):
    """根据观测对象自动初始化SVI参数"""
    atm_idx = np.argmin(np.abs(k_obs))#找最接近0的对数行权价索引
    w_atm = w_obs[atm_idx]#ATM点的波动率
    k_atm = k_obs[atm_idx]#ATM点的k值

    a_init = max(w_atm, 0.0001)
    m_init = k_atm

    k_left = k_obs.min()
    k_right = k_obs.max()
    w_left = w_obs[k_obs == k_left][0] if len(k_obs[k_obs == k_left]) > 0 else w_atm#这里w_obs[k_obs==k_left]是布尔索引的用法
    w_right = w_obs[k_obs == k_right][0] if len(k_obs[k_obs == k_right]) > 0 else w_atm

    slope_estimate = max(abs(w_right - w_left) / (k_right - k_left + 1e-6), 0.1)#这里+1e-6是为了防止除以0
    b_init = min(slope_estimate, 1.0)
    left_slope = (w_atm - w_left) / (k_atm - k_left + 1e-6) if k_atm > k_left else 0
    right_slope = (w_right - w_atm) / (k_right - k_atm + 1e-6) if k_right > k_atm else 0

    if left_slope > right_slope: #rho-偏度就是左右不对称的程度，所以左偏就是取-，右偏就是取＋
        rho_init = -0.3
    else:
        rho_init = 0.3

    sigma_init = max(0.1, (k_right - k_left) / 4)

    return [a_init, b_init, rho_init, m_init, sigma_init]
#----------------------指定到期日拟合--------------------------
    """优化算法
    ├── 一阶方法（只用梯度）
    │   ├── 梯度下降
    │   └── 动量法、Adam
    等
    │
    ├── 二阶方法（用梯度 + 海森矩阵）
    │   ├── 纯牛顿法
    │   ├── 牛顿下山法 ← 这个是牛顿法的改进
    │   └── 高斯 - 牛顿法
    │
    └── 拟牛顿法（用梯度 + 近似海森矩阵）
    ├── DFP
    ├── BFGS 
    └── L - BFGS - B（有限内存 + 边界约束）"""
def fit_svi_for_expiry(df_grouped,expiry_date):
    subset = df_grouped[df_grouped['expiry_date'] == expiry_date].copy()#由于链式索引返回的是视图，所以需要单独加一个副本防止原df被修改
    subset = subset.sort_values('log_moneyness')
    if len(subset) < 5:
        print(f"  警告: {expiry_date} 只有 {len(subset)} 个数据点，跳过")
        return None
    k_obs = subset['log_moneyness'].values
    w_obs = subset['w'].values
    T = subset['T'].iloc[0]
    weights = get_weights(k_obs)
    init_params = initialize_svi_params(k_obs, w_obs)
    print(f"\n拟合 {expiry_date} (T={T:.5f})...")
    print(
        f"  初始参数: a={init_params[0]:.5f}, b={init_params[1]:.5f}, rho={init_params[2]:.3f}, m={init_params[3]:.5f}, sigma={init_params[4]:.3f}")
    """minimize是一个优化器，来源于scipy，擅长的功能是取一个函数最小值，就是一个优化器"""
    result = minimize(
        svi_loss,
        init_params,
        args=(k_obs, w_obs, weights),
        method='L-BFGS-B',#内存占用小，支持边界约束,SVI就用这个方法就可以了
        bounds=[
            (0.000001, 1.0),
            (0.01, 2.0),
            (-0.99, 0.99),
            (-1.0, 1.0),
            (0.01, 1.0)
        ],
        options={'maxiter': 50000, 'disp': False}#最大迭代次数为50000次，disp=False就是静默模式，安静的运行，不输出
    )
    if result.success:
        a, b, rho, m, sigma = result.x
        print(f"  拟合成功! 损失: {result.fun:.2e}")
        print(f"  最终参数: a={a:.5f}, b={b:.5f}, rho={rho:.3f}, m={m:.5f}, sigma={sigma:.3f}")

        return {
            'expiry': expiry_date,
            'T': T,
            'params': result.x,
            'success': True,
            'loss': result.fun,
            'k_obs': k_obs,
            'w_obs': w_obs,
            'weights': weights
        }
    else:
        print(f"  拟合失败: {result.message}")
        return None
#-------------可视化----------------------
def plot_svi_fit(fit_result,save_path=None):
    """绘制 SVI 拟合结果"""
    if fit_result is None:
        print("无拟合结果")
        return
    """
    1.提取拟合数据
    2.生成平滑曲线
    3.计算R_square(残差平方和/总平方和：RSS/TSS)(rss是残差平方和(观测值和估计值)，tss是总方差（总偏离程度），ess是离差平方和（估计值和平均值）)
    4.绘图
    """

    k_obs = fit_result['k_obs']
    w_obs = fit_result['w_obs']
    T = fit_result['T']
    a, b, rho, m, sigma = fit_result['params']

    k_smooth = np.linspace(k_obs.min() - 0.1, k_obs.max() + 0.1, 200)#np.linspace是创建等差数列
    w_smooth = svi_w(k_smooth, a, b, rho, m, sigma)

    w_pred = svi_w(k_obs, a, b, rho, m, sigma)
    ss_res = np.sum((w_obs - w_pred) ** 2)
    ss_tot = np.sum((w_obs - np.mean(w_obs)) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))#创建两个子图

    # 总方差图
    ax1 = axes[0]
    ax1.scatter(k_obs, w_obs, c='blue', alpha=0.6, label='观测值')
    ax1.plot(k_smooth, w_smooth, 'r-', linewidth=2, label='SVI 拟合')
    ax1.set_xlabel('log_moneyness (k)')
    ax1.set_ylabel('总方差 w = IV_square × T')
    ax1.set_title(f'{fit_result["expiry"]}\nR_square = {r2:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # IV 微笑图
    ax2 = axes[1]
    iv_obs = np.sqrt(w_obs / T)
    iv_smooth = np.sqrt(w_smooth / T)
    ax2.scatter(k_obs, iv_obs, c='blue', alpha=0.6, label='观测值')
    ax2.plot(k_smooth, iv_smooth, 'r-', linewidth=2, label='SVI 拟合')
    ax2.set_xlabel('log_moneyness (k)')
    ax2.set_ylabel('隐含波动率 (IV)')
    ax2.set_title('隐含波动率微笑')
    ax2.legend()#显示图例
    ax2.grid(True, alpha=0.3)#增加网格线

    plt.tight_layout()#自动调节子图间距
    if save_path:#注意，这里需要将保存图片放在plt.show()之前防止阻塞
        plt.savefig(save_path,dpi=300,bbox_inches='tight')
        print(f"图片已保存到{save_path}")
    plt.show()

    residuals = w_obs - w_pred#残差
    print(f"\n拟合统计:")
    print(f"  R-squared = {r2:.4f}")
    print(f"  RMSE = {np.sqrt(np.mean(residuals ** 2)):.6f}")
    print(f"  残差均值 = {np.mean(residuals):.6f}")
    print(f"  残差标准差 = {np.std(residuals):.6f}")



# ============================================================
# 9. 运行拟合
# ============================================================

# 查看有哪些到期日
print("\n可用的 BTC 到期日:")
for exp in grouped['expiry_date'].unique():
    subset = grouped[grouped['expiry_date'] == exp]
    print(f"  {exp} (T={subset['T'].iloc[0]:.5f}, 合约数={len(subset)})")

"""test_expiry='2026/6/26 16:00'
result=fit_svi_for_expiry(grouped, test_expiry)
save_path='result_plt_show.png'
if result:
    plot_svi_fit(result,save_path)"""


# ============================================================
# 运行四张图
# ===========================================================
"""spot_price= binance_fetcher.get_spot_price('BTC')
if spot_price is None:
    print("请先运行价格获取模块获取现货价格")
else:
    # 图1和图2：plot_svi_fit
    test_expiry = sorted(grouped['expiry_date'].unique())[0]  # 取最近的到期日
    result = fit_svi_for_expiry(grouped, test_expiry)
    if result:
        plot_svi_fit(result, save_path='fig1_fig2_svi_fit.png')

    # 图3：多个到期日对比
    plot_multiple_tenors_iv(grouped, spot_price,
                            save_path='fig3_multiple_tenors.png')

    # 图4：3D曲面
    plot_vol_surface_3d(grouped, spot_price,
                        save_path='fig4_vol_surface.png')"""
spot_price = binance_fetcher.get_spot_price('BTC')
if spot_price is None:
    print("请先运行价格获取模块获取现货价格")
else:
    # 批量拟合所有到期日（预先拟合一次）
    print("\n开始批量拟合所有到期日...")
    all_fits = {}
    for expiry in grouped['expiry_date'].unique():
        subset = grouped[grouped['expiry_date'] == expiry]
        """T=subset['T'].iloc[0]
        if T < 0.0411:  # 15天 = 15/365 ≈ 0.0411年
            print(f"跳过 {expiry} (T={T:.5f}年 = {T * 365:.1f}天，期限太短)")
            continue"""
        if len(subset) >= 5:
            fit_result = fit_svi_for_expiry(grouped, expiry)
            if fit_result and fit_result['success']:
                all_fits[expiry] = fit_result
                print(f"✓ 成功拟合 {expiry}")

    # 图1和图2：单个到期日（取第一个成功拟合的）
    if all_fits:
        test_expiry = list(all_fits.keys())[0]
        result = all_fits[test_expiry]
        plot_svi_fit(result, save_path='fig1_fig2_svi_fit.png')

    # 图3：多个到期日对比
    if len(all_fits) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['blue', 'red', 'green', 'purple', 'orange']

        for i, (expiry, fit_result) in enumerate(list(all_fits.items())[:4]):
            T = fit_result['T']
            a, b, rho, m, sigma = fit_result['params']

            subset = grouped[grouped['expiry_date'] == expiry]
            k_obs = subset['log_moneyness'].values

            k_smooth = np.linspace(k_obs.min() - 0.1, k_obs.max() + 0.1, 200)
            w_smooth = svi_w(k_smooth, a, b, rho, m, sigma)
            iv_smooth = np.sqrt(w_smooth / T)

            ax.plot(k_smooth, iv_smooth, color=colors[i % len(colors)],
                    linewidth=2, label=f'{expiry} (T={T:.3f})')

        ax.set_xlabel('log_moneyness (k)')
        ax.set_ylabel('隐含波动率 (IV)')
        ax.set_title('多个到期日IV微笑对比')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('fig3_multiple_tenors.png', dpi=300, bbox_inches='tight')
        print("图3已保存到 fig3_multiple_tenors.png")
        plt.show()

    # 图4：3D曲面
    if len(all_fits) >= 2:
        from mpl_toolkits.mplot3d import Axes3D

        all_k, all_T, all_iv = [], [], []

        for expiry, fit_result in all_fits.items():
            T = fit_result['T']
            T_days=T*365
            a, b, rho, m, sigma = fit_result['params']

            subset = grouped[grouped['expiry_date'] == expiry]
            k_obs = subset['log_moneyness'].values

            k_smooth = np.linspace(k_obs.min() - 0.05, k_obs.max() + 0.05, 30)
            w_smooth = svi_w(k_smooth, a, b, rho, m, sigma)
            iv_smooth = np.sqrt(w_smooth / T)

            for k_val, iv_val in zip(k_smooth, iv_smooth):
                all_k.append(k_val)
                all_T.append(T_days)
                all_iv.append(iv_val)

        if all_k:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(all_k, all_T, all_iv, c=all_iv, cmap='coolwarm',
                                 s=20, alpha=0.7)

            ax.set_xlabel('log_moneyness (k)')
            ax.set_ylabel('到期时间 T (天)')
            ax.set_zlabel('隐含波动率 (IV)')
            ax.set_title('BTC期权隐含波动率曲面')

            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
            plt.tight_layout()
            plt.savefig('fig4_vol_surface.png', dpi=300, bbox_inches='tight')
            print("图4已保存到 fig4_vol_surface.png")
            plt.show()


