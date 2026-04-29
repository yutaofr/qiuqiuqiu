# 核心模型假设

1. 任何 `Adjusted Close` 必须是 **point-in-time adjusted**。禁止使用后向复权序列回灌历史。
2. 周 $t$ 的决策时间点只能使用该周最后一个已知观测之前的数据。禁止回写。
3. 所有协方差、平方根矩阵、求逆必须记录数值健康日志；失败则触发 `h_t^{state}=0`。
4. **工程越权声明：** 马氏距离计算使用谱分解托底后的 `cov_reg`（而非数学规范 §6.3 的 $\Sigma^{raw} + \epsilon I$）。谱分解托底仅提升次特征值、不膨胀主方向，在几何与数值稳定性上优于简单岭回归。数学规范 §6.3 的 $\epsilon_t I$ 视为"或等效的选择性谱分解托底"。
5. **协方差 NaN 延迟衰减语义：** 当 `RobustEWCov2D.update` 收到含 NaN 的二维输入时，`mean`、`cov_raw`、`cov_reg`、特征值和特征向量均保持不变，`warmup_count` 不增加，但 `pending_missing_steps` 增加 1。下一次有效输入到达时，递推使用 `effective_rho = rho ** (pending_missing_steps + 1)`，并用 `1 - effective_rho` 作为新观测的权重补集。更新成功后 `pending_missing_steps` 重置为 0。该行为等价于把缺失周的指数衰减延迟累计到下一次有效观测，避免用 NaN 或静默填充值污染状态。
6. **宏观价格契约边界：** `MacroMarketPriceContract` 仅允许用于 `state_stress_only` 诊断回放。它可以接收 `vendor_backward_adjusted`、`vendor_raw_close` 或 `official_market_close` 价格基准，不声明 PIT 微观层合规性。该契约禁止用于微观层、生产 `h_t` 和生产 `rho_t`；这些路径仍必须依赖 PIT 微观契约。
7. **FRED API ICE BofA 系列截断与 HYOAS CSV 覆盖策略：** FRED 免费 API 对 ICE BofA 系列（包括 `BAMLH0A0HYM2`）自约 2023 年 4 月起施加许可限制，仅提供约 3 年历史数据（~705 日行 → ~141 周行）。其余 7 个必需系列（美联储、芝加哥联储、CBOE 来源）不受影响，历史完整至 2000 年。这导致状态层双记忆因子无法完成 260 周预热，`build_replay_bundle()` 的 265 有限 theta 行门控失败，实时回放被锁定在 `"degraded"` 模式。

8. **Weekly Digest 编排约定：** `scripts/run_weekly_orchestration.sh` 的默认周目录在非 dry-run 时写入 `outputs/weekly/<week_end>`；`--dry-run` 未指定 `--output-dir` 时写入 `.temp/weekly/<week_end>`。`.env` 只能通过 allowlist loader 读取 `DISCORD_WEBHOOK_URL`，不得使用 `source .env`。

   **解决方案：** 在 `RealReplayConfig` 中设置 `hyoas_csv` 字段，指向包含长历史 `BAMLH0A0HYM2` 数据的 CSV 文件。CSV 格式要求：一个日期列（支持 `date`、`day`、`week_end`、`index` 等名称，或默认取第一列），一个数值列（支持 `BAMLH0A0HYM2`、`hyoas`、`hy_oas`、`hy_spread`、`value` 等名称，或默认取第二列）。列名大小写不敏感。建议时间跨度 ≥ 10 年（每日频率），以保证周重采样后有 ≥ 265 个有限 theta 行（建议原始日行 ≥ `MIN_WARMUP_ROWS = 525` 行）。

   **为什么不换用其他 HY 利差系列：** 替换为不同的利差序列（如 `BAMLH0A0HYM2EY` 或其他 OAS 变体）会改变 `compute_liquidity_factor()` 在整个历史上的数值结果，违反模型数学冻结约束（FF-2 边界）。`hyoas_csv` 必须包含原始 `BAMLH0A0HYM2` 数据（或经过文档记录的历史拼接版本）。

   **自动回归拼接（auto-splice）：** 当 FRED 返回的 `BAMLH0A0HYM2` 行数少于 `MIN_WARMUP_ROWS = 525` 且未提供 `hyoas_csv` 覆盖时，引擎自动调用 `_build_hyoas_splice()`：拉取 `BAA10Y`（美联储 Baa 公司债-国债利差，自 1962 年起免费提供），对已获取的 VIXCLS 和 NFCI 进行周频对齐，在 BAMLH0A0HYM2 实际数据的重叠窗口上校准 OLS 回归（BAA10Y, log(VIX), NFCI），将预测值填补至实际数据起点之前，与实际数据拼接后存入 `cache/real_replay/raw/hyoas_baa10y_splice.csv`。拼接方法记录在 manifest 的 `hyoas_splice_method` 字段（如 `"OLS(BAA10Y,lvix,nfci) R²=0.919 on 140w overlap"`）。`hyoas_source` 字段值为 `"baa10y_splice"`。

   **降级行为：** 显式 CSV 加载失败（文件不存在、列识别失败、日期范围内无数据）时，引擎记录 `source_errors["BAMLH0A0HYM2_csv_override"]` 并回退到 FRED API 路径（或 auto-splice）。auto-splice 失败时记录 `source_errors["BAMLH0A0HYM2_splice"]` 并保持 `hyoas_source = "fred_api"`。不进行静默替换。`series_coverage` 字段记录所有 8 个系列的原始日行数、最早日期和最晚日期，用于诊断未来的截断问题。
