# QQQ 周期状态识别系统 — 工程架构规范 v1.1

**数学基准件：** 模型规范 v2.2-final
**审计状态：** 代码级规格审计闭环。全部 Fatal Flaw 与 Material Weakness 已修正。两项 Noted Limitation 记录为单元测试覆盖项。
**放行门控：** (1) `transform_incremental` 的 NaN 必须被下游安全短路；(2) `get_adjusted_window` 在多重连环公司行为下累计因子浮点闭环精度 < $10^{-8}$。

---

## 1. 工程边界与禁止项

写入 `ASSUMPTIONS.md`：

1. 任何 `Adjusted Close` 必须是 **point-in-time adjusted**。禁止使用后向复权序列回灌历史。
2. 周 $t$ 的决策时间点只能使用该周最后一个已知观测之前的数据。禁止回写。
3. 所有协方差、平方根矩阵、求逆必须记录数值健康日志；失败则触发 `h_t^{state}=0`。
4. **工程越权声明：** 马氏距离计算使用谱分解托底后的 `cov_reg`（而非数学规范 §6.3 的 $\Sigma^{raw} + \epsilon I$）。谱分解托底仅提升次特征值、不膨胀主方向，在几何与数值稳定性上优于简单岭回归。数学规范 §6.3 的 $\epsilon_t I$ 视为"或等效的选择性谱分解托底"。

---

## 2. 仓库结构

```text
qqq_cycle/
  config/
    model_v22.yaml
  data_contracts/
    raw_prices.py
    corp_actions.py
    pit_adjustment.py       # v1.1 新增：滚动窗口点时复权引擎
    constituents.py
    weights.py
    fred_series.py
    exogenous_series.py
  core/
    calendar.py
    alignment.py
    robust_stats.py
    dual_memory.py
    covariance.py
    drift_probe.py          # v1.1 新增：§7.9 漂移探针
    rolling_quantile.py     # v1.1 新增：§8.3 动态噪声底
    proto_online.py
    state_layer.py
    stress_layer.py
    micro_layer.py
    risk_layer.py
    interpretability.py
  backtest/
    engine.py
    diagnostics.py
    oos_eval.py
  tests/
    test_alignment.py
    test_dual_memory.py
    test_covariance.py
    test_pit_adjustment.py  # v1.1 新增
    test_drift_probe.py     # v1.1 新增
    test_rolling_quantile.py # v1.1 新增
    test_proto_online.py
    test_micro_point_in_time.py
    test_oos_rules.py
```

---

## 3. 时间轴与数据契约

### 3.1 统一索引

```python
WeekIndex = pd.DatetimeIndex  # each item = decision week end timestamp
```

所有日频数据先对齐到交易日，再压到周频。

### 3.2 点时价格对象

```python
@dataclass(frozen=True)
class PITPriceBar:
    trade_date: pd.Timestamp
    ticker: str
    raw_close: float
    split_factor_cum_pti: float
    dividend_factor_cum_pti: float
    adj_close_pti: float
    asof_timestamp: pd.Timestamp
```

硬约束：

```python
adj_close_pti = raw_close * split_factor_cum_pti * dividend_factor_cum_pti
```

累计因子必须是 `trade_date` 当时可知的累计因子。

**使用范围限制（v1.1）：** `PITPriceBar` 仅用于单日决策快照（成员资格判断、当日收盘价读取）。**禁止**直接拼接历史 `adj_close_pti` 计算跨期滚动窗口。所有滚动计算必须通过 §3.4 的 `PITAdjustmentEngine`。

### 3.3 点时成员与权重对象

```python
@dataclass(frozen=True)
class PITConstituentSnapshot:
    trade_date: pd.Timestamp
    members: frozenset[str]
    weights: dict[str, float]  # sum close to 1.0
    asof_timestamp: pd.Timestamp
```

若当天没有权重但有成员：进入"降级严格模式"（数学规范 §2.3）。

### 3.4 点时滚动复权引擎（v1.1 新增）

```python
class PITAdjustmentEngine:
    """
    Provides corporate-action-adjusted prices with strict point-in-time semantics.
    All adjustments use only information knowable on or before `asof`.
    """

    def get_adj_close(
        self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> float:
        """
        Single-day adjusted close.
        Must raise DataNotAvailableError if only hindsight-adjusted series exists.
        """

    def get_adjusted_window(
        self, ticker: str, end_date: pd.Timestamp,
        window: int, asof: pd.Timestamp
    ) -> pd.Series:
        """
        Return `window`-length series ending at `end_date`, where every price
        is adjusted to the corporate action basis known as of `asof`.

        Implementation contract (relative basis scaling):
            P_adj(τ-k | asof) = P_raw(τ-k) * CUM_FAC(asof) / CUM_FAC(τ-k)

        where CUM_FAC(d) = split_factor_cum(d) * dividend_factor_cum(d)
        using only corporate actions known on or before `asof`.

        Raises:
            DataNotAvailableError: if PIT corporate action history unavailable
            InsufficientHistoryError: if fewer than `window` trading days available
        """
```

**微观层调用契约：** `micro_layer.py` 中 $MA_{20,i,\tau}^{adj}$ 的计算：

```python
window = pit_engine.get_adjusted_window(ticker, τ, 20, asof=τ)
ma_20 = window.mean()
```

如果 `PITAdjustmentEngine` 不可用（数据源不支持 `asof` 语义），微观层停机，系统降级为轻量模式。

---

## 4. 双记忆标准化模块

对应数学规范 §4。

### 4.1 接口定义

```python
class DualMemoryNormalizer:
    def __init__(
        self,
        robust_window: int,
        ew_half_life: int,
        eps: float = 1e-12,
        clip: tuple[float, float] | None = None,
        exo_var_huber_k: float | None = None,
    ) -> None: ...

    def fit_transform(self, x: pd.Series) -> pd.Series: ...
    def transform_incremental(self, x_new: float, history: pd.Series) -> float: ...
```

### 4.2 普通变量实现

```python
def z_rob(series: pd.Series, window: int, eps: float) -> pd.Series:
    med = series.shift(1).rolling(window, min_periods=window).median()
    mad = (series.shift(1) - med).abs().rolling(window, min_periods=window).median()
    scale = 1.4826 * mad + eps
    return (series - med) / scale
```

```python
def z_ew(series: pd.Series, half_life: int, eps: float) -> pd.Series:
    rho = 2 ** (-1 / half_life)
    n = len(series)
    mu = np.full(n, np.nan)
    var = np.full(n, np.nan)
    out = np.full(n, np.nan)

    MIN_EW_WARMUP = max(2, half_life // 4)

    mu[0] = series.iloc[0]
    var[0] = 0.0
    for i in range(1, n):
        x = series.iloc[i]
        mu[i] = rho * mu[i - 1] + (1 - rho) * x
        var[i] = rho * var[i - 1] + (1 - rho) * (x - mu[i - 1]) ** 2
        if i >= MIN_EW_WARMUP:
            out[i] = (x - mu[i]) / np.sqrt(var[i] + eps)
    # out[0:MIN_EW_WARMUP] remains NaN
    return pd.Series(out, index=series.index)
```

**NaN 策略（v1.1）：** `z_ew` 在 `t < MIN_EW_WARMUP` 期间返回 NaN。`dual_memory` 的输出在 `z_rob` 或 `z_ew` 任一为 NaN 时亦为 NaN。下游模块（§5 协方差、§state_layer 等）必须在接收到 NaN 特征向量时跳过该周的递推更新，不得静默填充。

```python
def dual_memory(series: pd.Series, robust_window: int, ew_half_life: int, eps: float) -> pd.Series:
    zr = z_rob(series, robust_window, eps)
    ze = z_ew(series, ew_half_life, eps)
    return 0.5 * zr + 0.5 * ze  # NaN propagates naturally
```

### 4.3 外生变量实现

```python
def exo_pretransform(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0.0))
```

```python
def z_ew_exo_with_huber_var(
    series: pd.Series, half_life: int, huber_k: float, eps: float
) -> pd.Series:
    rho = 2 ** (-1 / half_life)
    n = len(series)
    mu = np.full(n, np.nan)
    var = np.full(n, np.nan)
    out = np.full(n, np.nan)

    MIN_EW_WARMUP = max(2, half_life // 4)

    mu[0] = series.iloc[0]
    var[0] = 0.0
    for i in range(1, n):
        x = series.iloc[i]
        sigma_prev = np.sqrt(var[i - 1] + eps)
        delta = x - mu[i - 1]
        delta_clip = np.clip(delta, -huber_k * sigma_prev, huber_k * sigma_prev)
        mu[i] = rho * mu[i - 1] + (1 - rho) * x
        var[i] = rho * var[i - 1] + (1 - rho) * delta_clip ** 2
        if i >= MIN_EW_WARMUP:
            out[i] = (x - mu[i]) / np.sqrt(var[i] + eps)
    return pd.Series(out, index=series.index)
```

```python
def exo_dual_memory(series: pd.Series, eps: float) -> pd.Series:
    s = exo_pretransform(series)
    z1 = z_rob(s, window=260, eps=eps)
    z2 = z_ew_exo_with_huber_var(s, half_life=260, huber_k=4.0, eps=eps)
    return pd.Series(np.clip(0.5 * z1 + 0.5 * z2, -5.0, 5.0), index=series.index)
```

### 4.4 单元测试

```python
def test_no_future_in_robust_window():
    """today's output cannot change if future values are appended"""

def test_exo_clip_bounded():
    """exogenous score must stay within [-5, 5]"""

def test_exo_huber_var_not_polluted_by_single_50sigma_spike():
    """after one huge spike, EW variance should not remain exploded for 260 weeks"""

def test_dual_memory_nan_policy():
    """before enough history, output must be NaN and not silently imputed"""

def test_z_ew_warmup_nan():                          # v1.1
    """z_ew must return NaN for t < MIN_EW_WARMUP"""

def test_nan_does_not_propagate_to_covariance():      # v1.1
    """downstream covariance update must skip NaN inputs, not corrupt state"""
```

---

## 5. 协方差递推模块

对应数学规范 §6。

### 5.1 接口定义

```python
@dataclass
class CovarianceState2D:
    mean: np.ndarray          # shape (2,)
    cov_raw: np.ndarray       # shape (2, 2)
    cov_reg: np.ndarray       # shape (2, 2)
    eigvals: np.ndarray       # shape (2,)
    eigvecs: np.ndarray       # shape (2, 2)
    warmup_count: int
```

```python
class RobustEWCov2D:
    def __init__(
        self,
        half_life: int = 78,
        c_huber: float = 2.5,
        eps_abs: float = 1e-8,
        eps_rel: float = 1e-4,
        warmup_weeks: int = 260,
    ) -> None: ...

    def initialize_from_history(self, x_hist: np.ndarray) -> CovarianceState2D: ...
    def update(self, state: CovarianceState2D, x_t: np.ndarray) -> CovarianceState2D: ...
```

### 5.2 冷启动初始化（v1.1 新增）

```python
def initialize_from_history(self, x_hist: np.ndarray) -> CovarianceState2D:
    """
    x_hist: shape (N, 2), N >= 2.
    Uses static sample covariance + eps_abs * I as t=0 prior.

    Design constraint: initial prior is overwhelmed by EW recursion
    after ~3x half_life (~234 weeks), well within the 260-week warmup.
    warmup_count starts at 0; outputs remain locked until 260.
    """
    assert x_hist.shape[0] >= 2, "Need >= 2 observations for initialization"
    mean_init = x_hist.mean(axis=0)
    cov_sample = np.cov(x_hist.T, ddof=1)
    cov_raw_init = cov_sample + self.eps_abs * np.eye(2)
    cov_reg_init, eigvals, eigvecs = regularize_cov_2d(
        cov_raw_init, self.eps_abs, self.eps_rel
    )
    return CovarianceState2D(
        mean=mean_init,
        cov_raw=cov_raw_init,
        cov_reg=cov_reg_init,
        eigvals=eigvals,
        eigvecs=eigvecs,
        warmup_count=0,
    )
```

**数据断层处理（v1.1 NL-2）：** 若 `x_hist` 不足 2 条记录，`assert` 触发异常。调度层（`engine.py`）应捕获此异常并将对应模块标记为 `h_t^{state}=0`，而非全局崩溃。

### 5.3 谱分解与选择性托底

```python
def regularize_cov_2d(
    cov_raw: np.ndarray, eps_abs: float, eps_rel: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(cov_raw)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    lam1, lam2 = eigvals
    lam2_star = max(lam2, eps_rel * lam1, eps_abs)
    eigvals_star = np.array([lam1, lam2_star], dtype=float)
    cov_reg = eigvecs @ np.diag(eigvals_star) @ eigvecs.T
    return cov_reg, eigvals_star, eigvecs
```

**适用范围（数学规范 §6.7）：** 此函数用于所有 2D 协方差矩阵的正则化，包括 $\Sigma_{\Theta,t}$、$\Sigma_{v,t}$、$\Sigma_{a,t}$。

### 5.4 单步更新

```python
def update(self, state: CovarianceState2D, x_t: np.ndarray) -> CovarianceState2D:
    """
    NaN policy: if x_t contains NaN, return state unchanged (skip update).
    """
    if np.any(np.isnan(x_t)):
        return state

    rho = 2 ** (-1 / self.half_life)
    mean_prev = state.mean
    cov_reg_prev = state.cov_reg

    delta = x_t - mean_prev
    # Uses cov_reg (spectral floor) instead of cov_raw + εI (ridge).
    # See ASSUMPTIONS.md §4 for justification.
    maha = float(np.sqrt(delta.T @ np.linalg.inv(cov_reg_prev) @ delta))
    w = min(1.0, self.c_huber / max(maha, 1e-12))
    delta_tilde = w * delta

    mean_new = rho * mean_prev + (1 - rho) * x_t
    cov_raw_new = rho * state.cov_raw + (1 - rho) * np.outer(delta_tilde, delta_tilde)
    cov_reg_new, eigvals_new, eigvecs_new = regularize_cov_2d(
        cov_raw_new, self.eps_abs, self.eps_rel
    )

    return CovarianceState2D(
        mean=mean_new,
        cov_raw=cov_raw_new,
        cov_reg=cov_reg_new,
        eigvals=eigvals_new,
        eigvecs=eigvecs_new,
        warmup_count=state.warmup_count + 1,
    )
```

### 5.5 预热纪律

```python
def is_warm(self, state: CovarianceState2D) -> bool:
    return state.warmup_count >= self.warmup_weeks
```

在 `is_warm == False` 期间：禁止输出 $p_t$、$s_t$；禁止更新原型质心。

### 5.6 数值健康日志（v1.1 修正）

每次更新写出：

```python
{
    "date": ...,
    "maha": ...,
    "huber_weight": ...,
    "eigval_1": ...,
    "eigval_2_raw": ...,
    "eigval_2_reg": ...,
    "condition_number_raw": eigval_1 / max(eigval_2_raw, 1e-15),
    "condition_number_reg": eigval_1 / eigval_2_reg,
    "eigval_2_was_floored": eigval_2_raw < eps_rel * eigval_1,  # v1.1
    "state_ok": bool,
}
```

告警逻辑（v1.1 修正）：

```python
COND_WARN_RATIO = 0.9
cond_threshold = COND_WARN_RATIO / self.eps_rel  # 9000 at eps_rel=1e-4

if condition_number_reg > cond_threshold:
    # Secondary eigenvalue has been artificially floored;
    # matrix is approaching dimensional collapse.
    h_t_state = 1

if spectral_decomposition_failed or negative_eigenvalue or inverse_failed:
    h_t_state = 0
```

### 5.7 单元测试

```python
def test_cold_start_invertible():
    """cov_reg from initialize_from_history must be invertible"""

def test_warmup_gate():
    """is_warm returns False until 260 updates"""

def test_huber_truncation_bounds_outlier():
    """50-sigma outlier must be truncated, not allowed to corrupt covariance"""

def test_nan_input_skips_update():                    # v1.1
    """NaN input must leave state unchanged"""

def test_condition_threshold_reachable():              # v1.1
    """condition_number_reg must be able to reach COND_WARN_RATIO/eps_rel"""

def test_eigval_floored_flag():                        # v1.1
    """eigval_2_was_floored must be True when raw eigval < eps_rel * eigval_1"""
```

---

## 6. 第二批实现规格（接口预定义）

首批代码通过单元测试后启动。以下为接口骨架，详细实现规格在第二批审计中锁定。

### 6.1 漂移探针（数学规范 §7.9）

```python
# core/drift_probe.py

class RollingPercentile:
    """520-week rolling empirical percentile → [1/W, 1]"""
    def __init__(self, window: int = 520): ...
    def transform(self, series: pd.Series) -> pd.Series: ...

class DriftProbe:
    """
    Computes H_raw, H_bar_raw, Δ_abs_raw, α_drift, c_drift
    in physical (un-standardized) space.
    """
    def __init__(self, pct_window: int = 520, ew_half_life: int = 260,
                 theta_lo: float = 1.2, theta_hi: float = 1.8): ...
    def compute(self, raw_factors: dict[str, pd.Series]) -> pd.DataFrame: ...
```

### 6.2 动态噪声底（数学规范 §8.3）

```python
# core/rolling_quantile.py

def rolling_quantile_diag_2d(
    delta_v: np.ndarray,  # shape (T, 2)
    window: int = 520,
    quantile: float = 0.10,
) -> np.ndarray:
    """
    Returns shape (T, 2, 2) diagonal matrices.
    Uses expanding window for t < window.
    """
```

### 6.3 微观层断路器（数学规范 §9.11）

```python
# In micro_layer.py

def iir_envelope_with_breaker(
    h_t: float,
    h_t_minus1: float,
    h_t_minus2: float,
    x_lead_prev: float,
    delta: float = 0.9,
    theta_heal: float = 0.25,
) -> float:
    """
    Returns x_t_lead.
    Circuit breaker: if h_t < theta_heal for 3 consecutive weeks, x_lead = 0.
    """
```

### 6.4 连续混合带（数学规范 §10.1）

```python
# In risk_layer.py

def blended_state_weight(
    p_t: np.ndarray,           # shape (5,)
    omega_state: np.ndarray,   # shape (5,), fixed semantic weights
    delta_abs_raw: float,
    theta_lo: float = 1.2,
    theta_hi: float = 1.8,
    neutral: float = 0.6,
) -> float:
    """
    Returns ω_t with continuous linear blending between theta_lo and theta_hi.
    """
```

---

## 7. 首批回测：分布回放

在 PIT 复权和 PIT 权重不可用之前，不做收益回测。

### 7.1 模块级分布检查

2000–2025 周频分布统计（分位数 1%, 5%, 50%, 95%, 99%）：

- $\tilde g_t$, $s_t$, $d_t$, $a_t$
- $\Delta_t^{abs,raw}$
- $c_t^{drift}$ 触发周数
- 严格模式下 $h_t$, $h_t^{lead}$, $\rho_t$

### 7.2 事件对齐检查

三个已知窗口：2008-09→2009-06、2020-02→2020-06、2021-10→2022-03。

检验项：

- $\Delta_t^{abs,raw}$ 是否在窗口附近触发
- $p_t$ 是否向低热度簇迁移
- $s_t$ 是否在窗口内明显抬升
- 有微观层时 $\rho_t$ 与 $s_t$ 错位 ≤ 8 周

判废标准已在数学规范 §14.5 写定。

---

## 8. 实现顺序

**首批（可独立审计）：**

1. `calendar.py`, `alignment.py`
2. `dual_memory.py`（含 NaN 策略）
3. `covariance.py`（含冷启动、健康日志）
4. `state_layer.py` 中的 $L_t, T_t, P_t, E_t, \Theta_t$
5. `proto_online.py`
6. `stress_layer.py`

**第二批（需 PIT 数据或首批通过）：**

1. `pit_adjustment.py`（含 `get_adjusted_window`）
2. `drift_probe.py`, `rolling_quantile.py`
3. `constituents.py`, `weights.py`, `raw_prices.py`, `corp_actions.py`
4. `micro_layer.py`（含断路器）
5. `risk_layer.py`（含混合带）

---

## 9. 审计追溯

| # | 问题 | 严重性 | 来源 | 修正 | 状态 |
|---|------|--------|------|------|------|
| 1 | PIT 复权不支持滚动 MA | FF | 第三方 | `get_adjusted_window` 接口 | ✅ 已关闭 |
| 2 | 条件数阈值死代码 | FF | 第三方 | 相对阈值 + raw cond log | ✅ 已关闭 |
| 3 | 冷启动未定义 | MW | 第三方 | `initialize_from_history` | ✅ 已关闭 |
| 4 | maha 用 cov_reg vs raw+εI | NL | 第三方 | 接受越权，写入 ASSUMPTIONS | ✅ 已关闭 |
| 5 | v2.2 新增模块未覆盖 | MW | 自主 | 第二批接口预定义 | ✅ 已关闭 |
| 6 | z_ew 冷启动 var=0 | MW | 自主 | MIN_EW_WARMUP + NaN 策略 | ✅ 已关闭 |
| 7 | 仓库结构缺 drift_probe | NL | 自主 | 增加模块 | ✅ 已关闭 |
| 8 | NaN 连环污染 | NL | 终审 | update() NaN 短路 + 测试 | 🔵 监控 |
| 9 | 冷启动数据断层 | NL | 终审 | 调度层异常捕获 | 🔵 监控 |

**结论：首批实现（§4 标准化 + §6 协方差）的代码级规格已闭环，可进入编码。**
