# 2026-05-02 QQQ 周期编排系统 Gemini 提示词工程改进设计

## 1. 背景与目标
目前 Gemini 对 QQQ 周期状态识别系统的解读存在“语义漂移”，未能充分体现 SRD (Software Requirements Document) 中对各变量物理含义的设定。
本方案旨在通过“理论注入（Theory Grounding）”和“双轨叙事（Dual Narrative）”架构，提升解读的专业性、逻辑推断的准确性以及最终结论的可读性。

## 2. 核心设计

### 2.1 理论上下文注入 (System Theory Context)
在提示词中显式定义核心变量的物理内涵：
- **s_t (State Stress)**: 宏观动能强度。高值代表趋势一致性极高。
- **h_t (Micro Fragility)**: 微观脆弱性。衡量局部过度拥挤，与宏观动能解耦。
- **rho_t (Fracture Risk)**: 系统断裂风险。反映资产相关性同步共振，衡量流动性压力。
- **k_hat_t (Cycle Regime)**: 定义 1-5 对应的市场象限（早期、扩张、过热、回撤、出清）。

### 2.2 推演逻辑要求
- 要求 AI 结合 `execution_allowed` 与 `strict_gate_passed` 评估信号的确定性。
- 禁止空洞的宏观臆测，所有推论必须锚定在给定的数值波动上。

### 2.3 双轨叙事结构 (Dual Narrative)
1. **[核心推演]**: 专业量化解读，保留技术术语，引用数值。
2. **[理财顾问总结]**: 风格切换为“私人理财顾问”，禁止使用变量名和术语，使用生活化类比（如交通、气候、体力等）解释当前处境。

## 3. 验收标准
1. **逻辑对齐**: AI 对 k_hat=4 的解读必须包含“回撤”或“调整”的含义，而非随意发挥。
2. **符号清理**: 所有的 LaTeX 符号（如 $s_t$）必须在发送前被剥离（由已有的 sanitization logic 处理）。
3. **语言对齐**: 全程中文输出。

## 4. 实施路径
1. 更新 `scripts/run_weekly_orchestration.sh` 中的 `prompt` 定义。
2. 保持 `src/output/send_insight.py` 中的清理逻辑。
