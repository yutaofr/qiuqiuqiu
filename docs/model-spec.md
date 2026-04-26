# QQQ 周期状态识别系统完整模型规范 v2.2-final

**版本历史：** v2.0 → v2.1-final（状态应力方向中性化、语义标签分层、微观层权重缺失阈值、加速度协方差噪声底）→ v2.2-final（漂移探针解耦至物理空间、IIR 断路器、动态噪声底、连续混合权重降级）

**审计状态：** 终审三读闭环。全部 Fatal Flaw 与 Material Weakness 已修正。两项 Noted Limitation 记录为样本外监控项。

**样本外判废标准：** 若 $\Delta_t^{abs,raw}$ 报警序列与主状态分类在已知宏观拐点产生 >8 周相位错配，或常态震荡市中 $\rho_t$ 假阳性率 >30%，则系统判废，须重新约束 520 周窗口常数与经验超参数。

---

## 1. 系统定义

系统输出为：

$$
\mathcal{R}_t^{v2.2} = (\hat k_t,\ p_t,\ s_t,\ h_t,\ \rho_t,\ \mathcal{I}_t)
$$

其中：

- $\hat k_t$：主导周期状态
- $p_t$：五状态软概率
- $s_t$：状态应力（方向中性）
- $h_t$：独立微观结构脆弱性
- $\rho_t$：操作性断裂风险分数
- $\mathcal{I}_t$：审计型可解释性对象

硬约束：

$$
\rho_t \in [0,1], \qquad \rho_t \text{ 不是概率}
$$

$$
\text{若无历史真实成员矩阵，则 } h_t = \varnothing,\ \rho_t = \varnothing
$$

---

## 2. 运行模式

### 2.1 严格模式

要求从起始日 $t_0$ 开始，**每日归档**：

- 真实成分集合 $U_\tau$
- 成分股 Adjusted Close
- QQQ 官方披露的日度持仓权重向量 $w_\tau$

且满足：

$$
t \ge t_0 + 60 \text{ 个交易日}
$$

才允许输出完整的 $h_t, \rho_t$。

### 2.2 轻量模式

若没有历史成员矩阵，则：

$$
h_t = \varnothing, \qquad \rho_t = \varnothing
$$

系统仅输出：

$$
(\hat k_t,\ p_t,\ s_t,\ \mathcal{I}_t)
$$

### 2.3 降级严格模式

若有历史成员矩阵与 Adjusted Close，但没有历史权重向量 $w_\tau$，则：

- 广度分量可用
- 相关性集中分量停用
- $h_t$ 仅由广度构成
- 强制设置 $h_t^{micro} = 1$

---

## 3. 数据层

核心数据流固定为：

- DFII10
- DGS2
- BAMLH0A0HYM2
- NFCI
- VIXCLS
- USEPUINDXD
- AI-GPR
- QQQ 周价格
- 严格模式下的 $U_\tau$、Adjusted Close、$w_\tau$

删除项保持不变：

$$
\{\text{ETF持仓权重集中度},\ \Delta_{13}\log(NDX/NDXE),\ Q_t^{(h)}\}
$$

删除 $\Delta_{13}\log(NDX/NDXE)$ 的原因不变：NDXE 使用与 Nasdaq-100 相同证券并按季度再平衡，13 周差分与规则周期共振。

---

## 4. 标准化

短记忆稳健 z-score：

$$
z^{rob}_{w,t}(x) = \frac{x_t - \operatorname{median}(x_{t-w:t-1})}{1.4826 \cdot \operatorname{MAD}(x_{t-w:t-1}) + \varepsilon}
$$

长记忆 EW z-score：

$$
\rho_h = 2^{-1/h}
$$

$$
\mu^{EW,h}_t = \rho_h \mu^{EW,h}_{t-1} + (1 - \rho_h) x_t
$$

$$
v^{EW,h}_t = \rho_h v^{EW,h}_{t-1} + (1 - \rho_h)(x_t - \mu^{EW,h}_{t-1})^2
$$

$$
z^{EW,h}_t(x) = \frac{x_t - \mu^{EW,h}_t}{\sqrt{v^{EW,h}_t + \varepsilon}}
$$

普通双记忆标准化：

$$
\mathcal{S}_t(x) = 0.5\, z^{rob}_{104,t}(x) + 0.5\, z^{EW,260}_t(x)
$$

### 4.1 外生变量专用处理

对外生变量先做对数前变换：

$$
e_{1,t} = \log(1 + \text{AI-GPR}_t), \qquad e_{2,t} = \log(1 + \text{USEPUINDXD}_t)
$$

对其 EW 长记忆方差递推施加独立 Huber 截尾，仅作用于方差：

$$
\tilde\delta_{e,t}^{var} = \operatorname{clip}\big(e_t - \mu^{EW,260}_{t-1},\ -4\hat\sigma_{e,t-1},\ +4\hat\sigma_{e,t-1}\big)
$$

$$
v^{EW,260}_{e,t} = \rho_{260} v^{EW,260}_{e,t-1} + (1 - \rho_{260})(\tilde\delta_{e,t}^{var})^2
$$

然后定义：

$$
\mathcal{S}_t^{exo}(e) = \operatorname{clip}\!\left(0.5\, z^{rob}_{260,t}(e) + 0.5\, z^{EW,260}_t(e),\ -5,\ 5\right)
$$

---

## 5. 主状态层

流动性因子：

$$
L_t = \frac{1}{4}\Big[-\mathcal{S}_t(\text{DFII10}_t) - \mathcal{S}_t(\Delta_4 \text{DGS2}_t) - \mathcal{S}_t(\text{HYOAS}_t) - \mathcal{S}_t(\text{NFCI}_t)\Big]
$$

价格温度因子：

$$
u_{1,t} = \frac{QQQ_t}{MA_{52,t}} - 1, \qquad u_{2,t} = \frac{QQQ_t}{MA_{156,t}} - 1
$$

$$
T_t = \frac{1}{2}\mathcal{S}_t(u_{1,t}) + \frac{1}{2}\mathcal{S}_t(u_{2,t})
$$

风险偏好因子：

$$
P_t = \frac{1}{3}\Big[-\mathcal{S}_t(\text{VIXCLS}_t) - \mathcal{S}_t(\text{RV}_{20w,t}) + \mathcal{S}_t\!\left(\frac{QQQ_t}{MA_{40,t}} - 1\right)\Big]
$$

外生事件因子：

$$
E_t = \frac{1}{2}\mathcal{S}_t^{exo}(e_{1,t}) + \frac{1}{2}\mathcal{S}_t^{exo}(e_{2,t})
$$

状态坐标：

$$
\Theta_t = \begin{bmatrix} H_t \\ I_t \end{bmatrix}
$$

$$
H_t = 0.40\, L_t + 0.35\, T_t + 0.25\, P_t
$$

$$
I_t = 0.50\, \Delta_4 L_t + 0.30\, \Delta_4 T_t + 0.20\, \Delta_4 P_t
$$

---

## 6. 预热期与稳健协方差

### 6.1 预热期

系统要求 **260 周**历史预热。预热期内：

- 递推 $\bar\Theta_t, \Sigma_{\Theta,t}^{raw}$
- 不输出 $p_t, s_t$
- 不更新原型质心

### 6.2 均值递推

$$
\rho_\Sigma = 2^{-1/78}
$$

$$
\bar\Theta_t = \rho_\Sigma \bar\Theta_{t-1} + (1 - \rho_\Sigma)\Theta_t
$$

### 6.3 径向马氏 Huber 截断

$$
\delta_t = \Theta_t - \bar\Theta_{t-1}
$$

$$
m_t = \sqrt{\delta_t^\top (\Sigma^{raw}_{\Theta,t-1} + \epsilon_t I)^{-1} \delta_t}
$$

$$
w_t = \min\!\left(1, \frac{c_H}{m_t}\right), \qquad c_H = 2.5
$$

$$
\tilde\delta_t = w_t \delta_t
$$

### 6.4 协方差递推

$$
\Sigma^{raw}_{\Theta,t} = \rho_\Sigma \Sigma^{raw}_{\Theta,t-1} + (1 - \rho_\Sigma) \tilde\delta_t \tilde\delta_t^\top
$$

### 6.5 选择性特征值托底

对 $\Sigma^{raw}_{\Theta,t}$ 做对称谱分解：

$$
\Sigma^{raw}_{\Theta,t} = Q_t \begin{bmatrix} \lambda_{1,t} & 0 \\ 0 & \lambda_{2,t} \end{bmatrix} Q_t^\top, \qquad \lambda_{1,t} \ge \lambda_{2,t}
$$

$$
\lambda_{2,t}^* = \max(\lambda_{2,t},\ \epsilon_{rel}\lambda_{1,t},\ \epsilon_{abs})
$$

$$
\lambda_{1,t}^* = \lambda_{1,t}
$$

$$
\Sigma_{\Theta,t} = Q_t \begin{bmatrix} \lambda_{1,t}^* & 0 \\ 0 & \lambda_{2,t}^* \end{bmatrix} Q_t^\top
$$

建议：

$$
\epsilon_{abs} = 10^{-8}, \qquad \epsilon_{rel} = 10^{-4}
$$

### 6.6 平方根矩阵规范

所有 $\Sigma^{1/2}$ 与 $\Sigma^{-1/2}$ 一律使用对称谱分解，不允许用 Cholesky 因子代替坐标映射。

### 6.7 特征值托底的适用范围

第 6.5 节的选择性特征值托底（含 $\epsilon_{abs}, \epsilon_{rel}$）适用于本规范中所有协方差矩阵的谱分解，包括但不限于：

- $\Sigma_{\Theta,t}$（状态坐标协方差，第 6 节）
- $\Sigma_{v,t}$（速度协方差，第 8.2 节）
- $\Sigma_{a,t}$（加速度协方差，第 8.3 节）

---

## 7. 在线 EW 原型与状态概率

### 7.1 原型状态变量

对每个簇 $k$ 维护：

- 质心 $\mu_{k,t}$
- 有效权重 $W_{k,t}$
- 白化残差 $\xi_{k,t}$
- 最近一次非空时点 $t_k^*$

### 7.2 单周统一几何

本周所有分配、空簇映射与再激活判断，一律使用 $\Sigma_{\Theta,t-1}$。

### 7.3 当前点分配

$$
\hat k_t = \arg\min_k (\Theta_t - \mu_{k,t-1})^\top \Sigma_{\Theta,t-1}^{-1} (\Theta_t - \mu_{k,t-1})
$$

### 7.4 被分配簇的在线 EW 更新

设 $\rho_\mu = 2^{-1/78}$。若当前点分配到簇 $k = \hat k_t$，则：

$$
W_{k,t} = \rho_\mu W_{k,t-1} + 1
$$

$$
\mu_{k,t}^{raw} = \frac{\rho_\mu W_{k,t-1} \mu_{k,t-1} + \Theta_t}{\rho_\mu W_{k,t-1} + 1}
$$

其余簇：

$$
W_{j,t} = \rho_\mu W_{j,t-1}, \qquad j \neq k
$$

### 7.5 未分配簇：白化残差保留

$$
\mu_{j,t}^{raw} = \bar\Theta_t + \Sigma_{\Theta,t-1}^{1/2} \xi_{j,t_j^*}, \qquad j \neq \hat k_t
$$

### 7.6 再激活缓冲与闭式批更新

若某簇长期空缺后重新获得样本，设置 4 周缓冲。缓冲期结束时，设缓冲样本按远到近为 $\Theta_{t_1}, \Theta_{t_2}, \Theta_{t_3}, \Theta_{t_4}$，再激活前陈旧权重为 $W_{k,t_{last}^*}$，间隔为 $n_{gap}$ 周，则：

$$
W_k^{decayed} = \rho_\mu^{n_{gap}} W_{k,t_{last}^*}
$$

$$
\mu_k^{(5)} = \mu_{k,t-1}, \qquad W_k^{(5)} = W_k^{decayed}
$$

对 $j = 4, 3, 2, 1$ 递推：

$$
W_k^{(j)} = \rho_\mu W_k^{(j+1)} + 1
$$

$$
\mu_k^{(j)} = \frac{\rho_\mu W_k^{(j+1)} \mu_k^{(j+1)} + \Theta_{t_j}}{W_k^{(j)}}
$$

最终：

$$
\mu_{k,t}^{raw} = \mu_k^{(1)}, \qquad W_{k,t} = W_k^{(1)}
$$

### 7.7 白化残差存储

完成本周原型更新后，再用当期协方差存储白化残差：

$$
\xi_{k,t} = \Sigma_{\Theta,t}^{-1/2}(\mu_{k,t}^{raw} - \bar\Theta_t)
$$

并令：

$$
\mu_{k,t} = \mu_{k,t}^{raw}
$$

### 7.8 语义标签：相对语义 + 绝对漂移告警

按 $H$ 方向对五个质心排序。最低 $H$ 的两个簇中，$I$ 更低者标为 $S_1$，更高者标为 $S_2$；中位者为 $S_3$；最高 $H$ 的两个簇中，$I$ 更高者为 $S_4$，更低者为 $S_5$。

**但这些标签的宏观含义仅在"体制漂移告警关闭"时有效。** 漂移告警的触发由独立的物理空间漂移探针驱动（见第 7.9 节）。

### 7.9 物理空间漂移探针（v2.2 新增）

> 设计意图：主状态坐标 $\Theta_t$ 运行于滚动标准化空间（第 4–5 节），这保证了状态分类的稳健性，但同时消除了绝对体制漂移信息。漂移探针独立运行于物理空间，专用于检测标准化空间不可见的长周期体制平移。

#### 7.9.1 520 周滚动经验分位数映射

定义无量纲映射算子：

$$
\text{pct}_{520,t}(x) = \frac{\#\{x_j \le x_t : j \in [\max(1, t-520),\, t]\}}{W_t}
$$

其中 $W_t = \min(t, 520)$。$t < 520$ 时使用展开窗口（仅作冷启动过渡）。

该算子严格无量纲，输出 $\in [1/W_t,\, 1]$，保留单调序，且 520 周窗口避免了短记忆 z-score 的快速均值回归洗平。

#### 7.9.2 物理空间因子

$$
L_t^{raw} = \frac{1}{4}\Big[-\text{pct}_{520}(\text{DFII10}_t) - \text{pct}_{520}(\Delta_4 \text{DGS2}_t) - \text{pct}_{520}(\text{HYOAS}_t) - \text{pct}_{520}(\text{NFCI}_t)\Big]
$$

$$
T_t^{raw} = \frac{1}{2}\Big[\text{pct}_{520}(u_{1,t}) + \text{pct}_{520}(u_{2,t})\Big]
$$

$$
P_t^{raw} = \frac{1}{3}\Big[-\text{pct}_{520}(\text{VIXCLS}_t) - \text{pct}_{520}(\text{RV}_{20w,t}) + \text{pct}_{520}\!\left(\frac{QQQ_t}{MA_{40,t}} - 1\right)\Big]
$$

#### 7.9.3 物理空间健康坐标

$$
H_t^{raw} = 0.40\, L_t^{raw} + 0.35\, T_t^{raw} + 0.25\, P_t^{raw}
$$

$$
\bar H_t^{raw} = \rho_{260}^{EW}\, \bar H_{t-1}^{raw} + (1 - \rho_{260}^{EW})\, H_t^{raw}, \qquad \rho_{260}^{EW} = 2^{-1/260}
$$

#### 7.9.4 漂移度与告警

参考基线采用 520 周滚动中位数 + 520 周滚动 MAD：

$$
H_{med,t}^{520} = \text{rolling median}(H^{raw}_{[\max(1,t-520):t]})
$$

$$
\sigma_{H,t}^{520} = 1.4826 \cdot \text{rolling MAD}(H^{raw}_{[\max(1,t-520):t]})
$$

漂移度：

$$
\Delta_t^{abs,raw} = \frac{\bar H_t^{raw} - H_{med,t}^{520}}{\sigma_{H,t}^{520} + \varepsilon}
$$

漂移告警：

$$
c_t^{drift} = \mathbf{1}\{|\Delta_t^{abs,raw}| \ge \theta_{drift}^{hi}\}
$$

其中 $\theta_{drift}^{hi} = 1.8$（见第 10.1 节连续混合带定义）。

**可证伪验证：** 在 2000–2025 回测中，$\Delta_t^{abs,raw}$ 应在 2009 Q1 与 2021 Q4 附近各至少触发一次 $|\Delta_t^{abs,raw}| \ge 1.8$。若均未触发，首先检查窗口是否过长（降至 390 周），其次检查阈值是否过高。

### 7.10 状态概率

$$
D_{k,t}^2 = (\Theta_t - \mu_{k,t})^\top \Sigma_{\Theta,t}^{-1} (\Theta_t - \mu_{k,t})
$$

$$
p_{k,t} = \frac{\exp(-\tfrac{1}{2} D_{k,t}^2)}{\sum_{j=1}^{5} \exp(-\tfrac{1}{2} D_{j,t}^2)}
$$

$$
\hat k_t = \arg\max_k\, p_{k,t}
$$

---

## 8. 状态应力 $s_t$

删除 $\phi_t$。$s_t$ 恢复为方向中性的状态运动强度。

### 8.1 EMA 速度

$$
\rho_v = 2^{-1/4}
$$

$$
\Delta\Theta_t = \Theta_t - \Theta_{t-1}
$$

$$
v_t = \rho_v v_{t-1} + (1 - \rho_v) \Delta\Theta_t
$$

### 8.2 平滑位移强度

对 $v_t$ 做与第 6 节同样的稳健协方差递推（含第 6.5/6.7 节的特征值托底），得到 $\Sigma_{v,t}$。定义：

$$
d_t = \sqrt{v_t^\top \Sigma_{v,t}^{-1} v_t}
$$

### 8.3 趋势加速度与动态噪声底

定义：

$$
\delta^a_t = v_t - v_{t-1}
$$

对 $\delta^a_t$ 使用与第 6 节相同的径向马氏 Huber 截断，得到 $\tilde\delta^a_t$。

加速度协方差递推：

$$
\Sigma_{a,t} = \rho_\Sigma \Sigma_{a,t-1} + (1 - \rho_\Sigma) \tilde\delta^a_t \tilde\delta^{a\top}_t + \Sigma_{noise,t}
$$

**v2.2 修正：** $\Sigma_{noise,t}$ 改为 520 周滚动分位（替代 v2.1 的预热期末冻结）：

$$
\Sigma_{noise,t} = \operatorname{diag}\Big(Q_{0.10}\big[(\Delta v_H)^2_{[\max(1,t-520):t]}\big],\; Q_{0.10}\big[(\Delta v_I)^2_{[\max(1,t-520):t]}\big]\Big)
$$

$t < 520$ 时使用展开窗口 $1{:}t$。

$\Sigma_{a,t}$ 同样适用第 6.5/6.7 节的特征值托底。

然后定义：

$$
a_t = \sqrt{\delta^{a\top}_t \Sigma_{a,t}^{-1} \delta^a_t}
$$

### 8.4 应力输入

$$
g_t^{raw} = 0.5\, d_t + 0.5\, a_t
$$

$$
\tilde g_t = z^{rob}_{156,t}(g_t^{raw}), \qquad \tilde E_t = z^{rob}_{156,t}(E_t)
$$

$$
g_t^{stress} = 0.5\, \tilde g_t + 0.5\, \tilde E_t
$$

$$
s_t = \Lambda(g_t^{stress})
$$

---

## 9. 独立微观结构脆弱性 $h_t$

### 9.1 成员矩阵

$$
M_{i,\tau} = \begin{cases} 1, & i \in U_\tau \\ 0, & i \notin U_\tau \end{cases}
$$

### 9.2 Grace Period

对缺失不超过 3 个交易日的成员：

- 计数器冻结，不归零
- 不参与 $b_\tau, c_\tau$ 计算
- 当周强制设置 $c_t^{data} = 1$

### 9.3 熟化成员集合

$$
V_\tau^{20} = \{i : M_{i,\tau} = 1,\ A_{i,\tau} \ge 20\}
$$

$$
V_\tau^{60} = \{i : M_{i,\tau} = 1,\ A_{i,\tau} \ge 60\}
$$

### 9.4 价格契约

$$
P_{i,\tau} \equiv \text{Adjusted Close}_{i,\tau}
$$

### 9.5 巨头缺失权重阈值

定义 Grace Period 成员的前一交易日平滑滞后权重总和：

$$
w_{miss,\tau} = \sum_{i \in \text{Grace Period}_\tau} \bar w_{i,\tau^-}
$$

定义阈值：

$$
\theta_{miss,\tau} = 0.5 \times w_{(5),\tau^-}
$$

其中 $w_{(5),\tau^-}$ 为前一交易日第五大平滑滞后权重。

若：

$$
w_{miss,\tau} > \theta_{miss,\tau}
$$

则禁止当日微观重算：

$$
b_\tau = b_{\tau-1}, \qquad c_\tau = c_{\tau-1}, \qquad c_t^{data} = 1
$$

### 9.6 平滑滞后权重

设 5 个交易日半衰期：

$$
\rho_w = 2^{-1/5}
$$

若当日不处于规则窗口：

$$
\bar w_{\tau^-} = \rho_w \bar w_{(\tau-1)^-} + (1 - \rho_w) w_{\tau-1}
$$

若处于规则窗口，则：

$$
\bar w_{\tau^-} = \bar w_{(\tau-1)^-}
$$

### 9.7 广度坍缩

采用权重化广度：

$$
b_\tau = 1 - \frac{\sum_{i \in V_\tau^{20}} \bar w_{i,\tau^-} \mathbf{1}\{P_{i,\tau} > MA_{20,i,\tau}^{adj}\}}{\sum_{i \in V_\tau^{20}} \bar w_{i,\tau^-} + \varepsilon}
$$

### 9.8 加权相关性集中

定义限制在 $V_\tau^{60}$ 上并重新归一化的平滑滞后权重：

$$
\tilde{\mathbf{w}}_{\tau^-}^{60} = \frac{(\bar w_{i,\tau^-} \mathbf{1}\{i \in V_\tau^{60}\})_i}{\sum_{j \in V_\tau^{60}} \bar w_{j,\tau^-} + \varepsilon}
$$

定义 60 日相关矩阵 $R_\tau$。相关性集中度定义为归一化加权平均相关：

$$
c_\tau = \frac{\tilde{\mathbf{w}}_{\tau^-}^{60\top} (R_\tau - I) \tilde{\mathbf{w}}_{\tau^-}^{60}}{1 - |\tilde{\mathbf{w}}_{\tau^-}^{60}|_2^2 + \varepsilon}
$$

### 9.9 周频聚合

$$
b_t^{wk} = \operatorname{median}_{\tau \in week(t)} b_\tau, \qquad c_t^{wk} = \operatorname{median}_{\tau \in week(t)} c_\tau
$$

### 9.10 规则周降权

$$
\omega_\tau = \begin{cases} 0.3, & c_\tau^{rule} = 1 \text{ or } c_\tau^{const} = 1 \\ 1.0, & \text{otherwise} \end{cases}
$$

定义加权稳健 z-score：

$$
\tilde b_t = z^{wrob}_{156,t}(b_t^{wk}; \omega), \qquad \tilde c_t = z^{wrob}_{156,t}(c_t^{wk}; \omega)
$$

### 9.11 微观分数与 IIR 领先记忆

定义微观原始分数：

$$
M_t^{raw} = 0.5\, \tilde b_t + 0.5\, \tilde c_t
$$

先映射到有界空间：

$$
h_t = \Lambda(M_t^{raw})
$$

再做去中心化且**正区间记忆**的 IIR 包络。定义：

$$
x_t = (h_t - 0.5)_+
$$

**v2.2 修正：含断路器的 IIR 递推**

定义治愈断路器条件：

$$
\text{heal}_t = \mathbf{1}\{h_t < \theta_{heal} \text{ 且 } h_{t-1} < \theta_{heal} \text{ 且 } h_{t-2} < \theta_{heal}\}
$$

其中 $\theta_{heal} = 0.25$。

$$
x_t^{lead} = \begin{cases} 0, & \text{if } \text{heal}_t = 1 \\ \max(x_t,\ \delta\, x_{t-1}^{lead}), & \text{otherwise} \end{cases}
$$

$$
\delta = 0.9
$$

$$
h_t^{lead} = 0.5 + x_t^{lead}
$$

**设计意图：** 不引入单周硬重置（避免单周噪声误清），要求连续 3 周确认微观结构处于深度健康区，才切断记忆链。

**可证伪验证：** 在回测中定位"急速修复后二次下探"事件（如 2020-03 反弹后 2020-06 回调）。若断路器在首次反弹时过早清零导致 $\rho_t$ 未能捕捉二次下探，则 $N_{heal}$ 需上调或 $\theta_{heal}$ 需下调。

**样本外监控项（NL-2）：** $\theta_{heal} = 0.25$ 为 $\Lambda(\cdot)$ 输出空间的静态阈值。若长期回测表明该阈值在不同流动性体制间缺乏一致性，可替换为 $h_t$ 自身的长期滚动分位数。

---

## 10. 操作性断裂风险分数 $\rho_t$

### 10.1 状态条件权重（v2.2 修正：连续混合带）

定义固定语义权重：

$$
\omega^{state} = \{S_1: 1.0,\ S_2: 0.7,\ S_3: 0.3,\ S_4: 0.6,\ S_5: 0.9\}
$$

定义自然状态权重：

$$
\omega_t^{nat} = \sum_{k=1}^{5} p_{k,t}\, \omega_k^{state}
$$

定义漂移混合系数（替代 v2.1 的硬开关）：

$$
\alpha_t^{drift} = \operatorname{clip}\!\left(\frac{|\Delta_t^{abs,raw}| - \theta_{drift}^{lo}}{\theta_{drift}^{hi} - \theta_{drift}^{lo}},\ 0,\ 1\right)
$$

其中 $\theta_{drift}^{lo} = 1.2$，$\theta_{drift}^{hi} = 1.8$。

状态条件权重：

$$
\omega_t = (1 - \alpha_t^{drift})\, \omega_t^{nat} + \alpha_t^{drift} \cdot 0.6
$$

性质：

- $|\Delta_t^{abs,raw}| < 1.2$：$\omega_t = \omega_t^{nat}$，语义权重完全生效
- $1.2 \le |\Delta_t^{abs,raw}| \le 1.8$：平滑过渡
- $|\Delta_t^{abs,raw}| > 1.8$：$\omega_t = 0.6$，语义完全降级为中性

### 10.2 状态条件应力与微观放大

$$
m_t = \omega_t\, s_t
$$

$$
n_t = 2(h_t^{lead} - 0.5)_+ \in [0,1]
$$

### 10.3 最终风险分数

$$
\rho_t = 1 - (1 - m_t)(1 - \lambda_\rho\, n_t)
$$

固定：

$$
\lambda_\rho = 0.75
$$

性质：

- $\rho_t \in [0,1]$
- 对 $m_t, n_t$ 单调递增
- 不因相关性上升而机械折价
- 相同的 $s_t$ 在 $S_1/S_5$ 主导下比在 $S_3$ 主导下得到更高 $\rho_t$
- 语义失效时通过连续混合带平滑降级为中性状态权重
- $\omega_t$ 在漂移边界上连续，不产生伪跳跃

### 10.4 相关性只进入解释层

定义：

$$
\eta_t = \operatorname{EWCorr}_{78w}(s_t, h_t^{lead})
$$

$\eta_t$ 只进入 $\mathcal{I}_t$，不进入 $\rho_t$ 本体。

---

## 11. 审计型可解释性 $\mathcal{I}_t$

$$
\mathcal{I}_t = (\mathcal{A}_t,\ \mathcal{C}_t,\ \mathcal{D}_t,\ \mathcal{H}_t)
$$

### 11.1 归因

$$
\mathcal{A}_t^H = (0.40\, L_t,\ 0.35\, T_t,\ 0.25\, P_t)
$$

$$
\mathcal{A}_t^I = (0.50\, \Delta_4 L_t,\ 0.30\, \Delta_4 T_t,\ 0.20\, \Delta_4 P_t)
$$

$$
\mathcal{A}_t^s = (\tilde g_t,\ \tilde E_t)
$$

$$
\mathcal{A}_t^h = (\tilde b_t,\ \tilde c_t)
$$

$$
\mathcal{A}_t^\rho = (\omega_t\, s_t,\ \lambda_\rho\, n_t,\ \eta_t)
$$

### 11.2 污染标记

$$
\mathcal{C}_t = (c_t^{rule},\ c_t^{const},\ c_t^{data},\ c_t^{micro},\ c_t^{drift})
$$

$$
c_t^{rule} = \mathbf{1}\{t \in \mathcal{W}_{rebalance} \cup \mathcal{W}_{reconstitution} \cup \mathcal{W}_{fastentry} \cup \mathcal{W}_{replacement}\}
$$

$$
c_t^{const} = \mathbf{1}\{|U_t \triangle U_{t-4}| > 0\}
$$

$$
c_t^{data} = \mathbf{1}\{\text{核心时间序列缺失、权重前向填充、或触发 Grace Period}\}
$$

$$
c_t^{micro} = \mathbf{1}\!\left\{\frac{|V_t^{60}|}{|U_t|} < 0.80\right\}
$$

$$
c_t^{drift} = \mathbf{1}\{|\Delta_t^{abs,raw}| \ge \theta_{drift}^{hi}\}
$$

### 11.3 漂移诊断

$$
d_t^{state} = \sqrt{(\Theta_t - \bar\Theta_t)^\top \Sigma_{\Theta,t}^{-1} (\Theta_t - \bar\Theta_t)}
$$

$$
d_t^{stress} = |g_t^{stress}|
$$

$$
d_t^{frag} = |M_t^{raw}|
$$

$$
d_t^{abs} = |\Delta_t^{abs,raw}|
$$

### 11.4 模块健康度

$$
\mathcal{H}_t = (h_t^{macro},\ h_t^{exo},\ h_t^{micro},\ h_t^{state})
$$

- 若宏观核心序列缺失超过 1 项：$h_t^{macro} = 0$
- 若 AI-GPR 或 EPU 未更新：$h_t^{exo} = 1$，都未更新则 $0$
- 若无历史成员矩阵或无 Adjusted Close：$h_t^{micro} = 0$
- 若缺少历史权重向量但有成员矩阵与 Adjusted Close：$h_t^{micro} = 1$
- 若 $\Sigma_{\Theta,t}$ 或 $\Sigma_{a,t}$ 求逆失败：$h_t^{state} = 0$

---

## 12. 周频运行顺序

1. 更新 FRED 与 AI-GPR 序列
2. 更新 QQQ 周价格
3. 严格模式下更新每日成员矩阵、Adjusted Close、权重向量
4. 计算 $L_t, T_t, P_t, E_t$
5. 若仍在 260 周预热期：仅更新 $\bar\Theta_t, \Sigma_{\Theta,t}^{raw}$ 与漂移探针原始序列，停止
6. 预热结束后初始化在线原型状态 $(\mu_k, W_k, \xi_k)$
7. 更新稳健协方差
8. 用 $\Sigma_{\Theta,t-1}$ 执行当前点分配与在线 EW 原型更新
9. 计算 $p_t, \hat k_t$
10. 计算漂移探针：$H_t^{raw}, \bar H_t^{raw}, \Delta_t^{abs,raw}, \alpha_t^{drift}, c_t^{drift}$
11. 计算 $v_t, d_t, a_t, g_t^{stress}, s_t$
12. 计算 $\omega_t$（含连续混合带）
13. 严格模式下计算 $b_\tau, c_\tau, \tilde b_t, \tilde c_t, h_t, h_t^{lead}$（含断路器），计算 $\rho_t$
14. 轻量模式下：$h_t = \varnothing,\ \rho_t = \varnothing$
15. 输出 $\mathcal{I}_t$

---

## 13. 已知限制与样本外监控项

### 13.1 次要问题

**Material Weakness（已关闭）：**

1. 在线 EW 原型更新牺牲了部分离线聚类最优性，换来单周定义闭合与数值稳定。
2. $c_\tau$ 现在依赖历史权重向量 $w_\tau$，这提高了严格模式的数据要求。
3. $\rho_t$ 使用固定 $\lambda_\rho$ 和固定 $\omega^{state}$ 先验。这是有意保守，不是统计最优。

**Noted Limitation（开放监控）：**

- **NL-1：分位数极限饱和。** 当输入变量呈现跨度远超 520 周的单边物理运动时，$\text{pct}_{520,t}$ 会在长达数年的时间内输出 $0$ 或 $1$ 的饱和值，该变量在饱和期间丧失边际变化信息。代数上安全但信息量退化。
- **NL-2：静态治愈阈值。** $\theta_{heal} = 0.25$ 假设微观分数四分位边界在不同市场体制间具有相同的物理修复语义。若长期回测证伪，可替换为 $h_t$ 的长期滚动分位数。

### 13.2 Noted Limitation（结构性）

QQQ 官方页面说明持仓会变化，而 Nasdaq-100 的 2026 生效版方法学明确存在 Fast Entry、季度再平衡、年度重构与替换规则；因此，严格模式的 $h_t$ 必须依赖用户自己的历史归档，而不是单日网页快照。

---

## 14. 评估变更标准

1. 若 $\Delta_t^{abs,raw}$ 在 2009 Q1 与 2021 Q4 均未触发告警，则漂移探针参数需重新校准（优先检查 520 周窗口长度，其次检查 $\theta_{drift}^{hi}$）。
2. 若匀速期后单次极小扰动仍让 $a_t$ 爆炸，说明 520 周滚动 $\Sigma_{noise,t}$ 窗口过长或分位过低，应缩短窗口或上调分位。
3. 若长期平静期里 $h_t^{lead}$ 仍被机械抬高，检查是否误把负区间纳入包络递推或断路器未被正确实现。
4. 若缺少历史成员矩阵、Adjusted Close 或权重向量，则当前唯一完整合法的输出仍然是 $(\hat k_t,\ p_t,\ s_t,\ \mathcal{I}_t)$，而 $h_t, \rho_t$ 只能处于关闭或降级状态。
5. **样本外判废标准：** 若 $\Delta_t^{abs,raw}$ 报警序列与主状态分类在已知宏观拐点产生 >8 周相位错配，或常态震荡市中 $\rho_t$ 假阳性率 >30%，则系统判废，须重新约束 520 周窗口常数与各类经验超参数的自由度。

---

## 15. 审计追溯

| 轮次 | 问题 | 严重性 | 修正 | 状态 |
|------|------|--------|------|------|
| 三读 | 漂移告警在标准化空间失效 | Fatal Flaw | 解耦至物理空间独立探针 | ✅ 已关闭 |
| 四读 | 展开统计量渐近发散 | Fatal Flaw | 520w 滚动替换展开 | ✅ 已关闭 |
| 四读 | $P_t^{raw}$ 量纲支配 | Material Weakness | 520w 滚动分位数映射 | ✅ 已关闭 |
| 三读 | IIR 治愈阻断 | Material Weakness | 3 周连续健康断路器 | ✅ 已关闭 |
| 三读 | $\Sigma_{noise}$ 预热期冻结 | Material Weakness | 520w 滚动分位 | ✅ 已关闭 |
| 三读 | $\omega_t$ 阶跃不连续 | Material Weakness | 线性混合带 $[1.2, 1.8]$ | ✅ 已关闭 |
| 三读 | $\Sigma_{v,t}$ 缺乏特征值托底 | Noted Limitation | 显式引用 §6.5 托底 | ✅ 已关闭 |
| 三读 | 权重归一化 Grace Period | Noted Limitation | 驳回（重归一化已存在） | ✅ 已关闭 |
| 终审 | 分位数极限饱和 (NL-1) | Noted Limitation | 样本外监控 | 🔵 开放 |
| 终审 | 静态治愈阈值 (NL-2) | Noted Limitation | 样本外监控 | 🔵 开放 |

**终审结论：v2.2-final 的数学架构在封闭代数系统内逻辑自洽，全部已知致命缺陷与实质性弱点已修正，具备执行代码落地的严密性。**
