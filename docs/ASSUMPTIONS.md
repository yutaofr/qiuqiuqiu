# 核心模型假设

1. 任何 `Adjusted Close` 必须是 **point-in-time adjusted**。禁止使用后向复权序列回灌历史。
2. 周 $t$ 的决策时间点只能使用该周最后一个已知观测之前的数据。禁止回写。
3. 所有协方差、平方根矩阵、求逆必须记录数值健康日志；失败则触发 `h_t^{state}=0`。
4. **工程越权声明：** 马氏距离计算使用谱分解托底后的 `cov_reg`（而非数学规范 §6.3 的 $\Sigma^{raw} + \epsilon I$）。谱分解托底仅提升次特征值、不膨胀主方向，在几何与数值稳定性上优于简单岭回归。数学规范 §6.3 的 $\epsilon_t I$ 视为"或等效的选择性谱分解托底"。
5. **协方差 NaN 延迟衰减语义：** 当 `RobustEWCov2D.update` 收到含 NaN 的二维输入时，`mean`、`cov_raw`、`cov_reg`、特征值和特征向量均保持不变，`warmup_count` 不增加，但 `pending_missing_steps` 增加 1。下一次有效输入到达时，递推使用 `effective_rho = rho ** (pending_missing_steps + 1)`，并用 `1 - effective_rho` 作为新观测的权重补集。更新成功后 `pending_missing_steps` 重置为 0。该行为等价于把缺失周的指数衰减延迟累计到下一次有效观测，避免用 NaN 或静默填充值污染状态。
6. **宏观价格契约边界：** `MacroMarketPriceContract` 仅允许用于 `state_stress_only` 诊断回放。它可以接收 `vendor_backward_adjusted`、`vendor_raw_close` 或 `official_market_close` 价格基准，不声明 PIT 微观层合规性。该契约禁止用于微观层、生产 `h_t` 和生产 `rho_t`；这些路径仍必须依赖 PIT 微观契约。
