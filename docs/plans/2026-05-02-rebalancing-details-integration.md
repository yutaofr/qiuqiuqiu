# 2026-05-02 QQQ 周期编排系统：调仓细节集成设计

## 1. 背景与目标
目前 Discord 报告仅显示“调仓数量”，缺乏具体的比例变化。用户希望看到权重增减（Weight Delta），以便直观把握系统的防御性动作。

## 2. 核心设计

### 2.1 数据聚合 (Data Aggregation)
- **src/main.py**: 修改 `_phase15_section`。
- **逻辑**: 根据 `week_end` 寻找 `outputs/phase15/portfolio_delta_{week_end}.json`。
- **提取**: 将 `delta_weights` 字段注入 `weekly_report.json`。

### 2.2 数据脱敏 (Sanitization)
- **scripts/sanitize_weekly_report.py**: 
- **逻辑**: 在 `_sanitize_phase15` 的 `allowed` 白名单中增加 `delta_weights`。

### 2.3 提示词工程 (Prompt Engineering)
- **上下文**: 告知 Gemini `delta_weights` 的含义（0.1 = 10% 权重变化）。
- **任务**: 
    - 技术层：分析调仓的方向（如“向避险资产 BIL 转移”）。
    - 顾问层：将比例变化转化为生活化表述（如“撤回四成兵力”、“补充了三成备用金”）。

## 3. 验收标准
1. **数据呈现**: 周报 JSON 包含 `delta_weights: {"QQQ": -0.4, "BIL": 0.4}`。
2. **AI 解读**: Discord 报告中明确提及 40% 的权重调整逻辑。
3. **安全合规**: 不暴露任何本地路径。

## 4. 实施路径
1. 依次修改 `src/main.py`, `scripts/sanitize_weekly_report.py`。
2. 更新 `scripts/run_weekly_orchestration.sh` 中的提示词模板。
3. 运行重发测试验证全链路。
