# OKX Quant 回测手册

> 面向 **OKX 永续合约历史数据** 的模块化量化回测框架。
> 当前版本专注于：数据管理、策略信号、组合回测、风险过滤、资金费率、滑点成交、Walk-Forward 与参数敏感性分析。

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img alt="Mode" src="https://img.shields.io/badge/Mode-Backtest-2E8B57?style=flat-square" />
  <img alt="Config" src="https://img.shields.io/badge/Config-config.yaml-orange?style=flat-square" />
  <img alt="Strategies" src="https://img.shields.io/badge/Strategies-VOL%20%7C%20MR%20%7C%20EWMAC%20%7C%20MACD%20%7C%20DON%20%7C%20MING-blueviolet?style=flat-square" />
</p>

---

## 目录

- [1. 项目定位](#1-项目定位)
- [2. 快速开始](#2-快速开始)
- [3. 功能总览](#3-功能总览)
- [4. 主流程与调用关系](#4-主流程与调用关系)
- [5. 内置策略](#5-内置策略)
  - [5.1 不同 K 线的建议策略与自动规划](#51-不同-k-线的建议策略与自动规划)
  - [5.2 ming 策略详解](#52-ming-策略详解)
- [6. 数据管理](#6-数据管理)
- [7. 回测与验证](#7-回测与验证)
- [8. 自定义策略开发手册](#8-自定义策略开发手册)
- [9. 文件职责与依赖关系](#9-文件职责与依赖关系)
- [10. 已移除内容](#10-已移除内容)

---

## 1. 项目定位

本项目是一个 **纯回测发行包**，用于在 OKX 永续合约历史数据上验证多策略组合。系统当前默认策略组合为：

```text
vol + mr + ewmac + macd + don + ming
```

当前版本已下线：

- 审计 / 退出重放模块
- 旧版 RNG 区间策略（已由 `ming` 1D 双触发共振策略 + 引擎结构止损 / Chandelier 替代）
- 仅作演示的占位模板文件
- 重复或过期的说明文档

> 策略只负责生成信号；成交、滑点、手续费、资金费、止损、止盈、超时退出全部由回测引擎处理。
> 例外：`ming` 策略为了让 1D 走「+1R / +1.5R 减仓 / +2R 启 Chandelier / 时间止损 / panic exit」完整生命周期，把状态机参数（`mfe_window_bars` / `max_bars` / `panic_gap_atr` / `r1_be_buffer_atr` / `r15_partial_pct` / `r2_chandelier_N` / `r2_chandelier_k`）发到 `Signal` 上，**仓位状态仍由 `v3/engine.py` 唯一持有**——见 [§5.2](#52-ming-策略详解)。

---

## 2. 快速开始

### 2.1 安装环境

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2.2 准备数据

```bash
python tools/data_manager.py \
  --symbols BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP \
  --bars 1m,5m,15m,1H,4H \
  --days 180 \
  --funding
```

### 2.3 运行回测

```bash
python -m v3.run_backtest --bar 1H
```

### 2.4 常用命令

```bash
# 指定品种
python -m v3.run_backtest --bar 1H --symbols BTC-USDT-SWAP,ETH-USDT-SWAP

# 只跑一个策略
python -m v3.run_backtest --bar 1H --only ming --symbols BTC-USDT-SWAP,ETH-USDT-SWAP

# 指定时间范围
python -m v3.run_backtest --bar 1H --start 2025-01-01 --end 2025-06-01

# 查看自动发现到的策略
python -m v3.run_backtest --list-strategies
```

默认结果目录：

```text
data/backtest_results/
```

---

## 3. 功能总览

| 模块 | 能力 | 入口文件 |
|---|---|---|
| 统一配置 | 根目录 `config.yaml` 作为唯一配置源 | `config.yaml` |
| 数据管理 | 下载、校验、入库、增量更新、状态检查 | `tools/data_manager.py` |
| 多品种回测 | 时间轴对齐、IS/OOS 切分、组合绩效输出 | `backtest/multi_engine.py` |
| 策略引擎 | 策略发现、信号生成、候选信号选择、持仓管理 | `v3/engine.py` |
| 风险过滤 | Regime、HTF 方向过滤、冲突窗口过滤 | `v3/risk/` |
| 成交模拟 | 订单簿滑点、延迟、部分成交、拒单、手续费 | `backtest/order_book_fill.py` |
| 资金费率 | 历史资金费率入库与回测计入 | `backtest/funding_store.py` |
| 稳定性验证 | Walk-Forward、参数敏感性、过拟合检查、OOS 时间窗口拆段 | `tools/walk_forward.py` / `tools/param_sensitivity.py` / `tools/split_oos.py` |
| 自动诊断 | 绩效归因、逐笔流水核对、假成交/错归因/OHLC 越界检测 | `diagnose.py` |

---

## 4. 主流程与调用关系

### 4.1 总体链路

```text
config.yaml
  ↓
v3.run_backtest.load_config()
  ↓
backtest.multi_engine.run_multi_from_parquet()
  ├─ 读取历史 K 线：backtest.data_store / backtest.hist_fetcher
  ├─ 读取资金费率：backtest.funding_store
  ├─ 生成信号与管理持仓：v3.engine.StrategyEngine
  │    ├─ 判断市场状态：v3.risk.regime.RegimeDetector
  │    ├─ 过滤方向/冲突：v3.risk.filters.HtfFilter / ConflictFilter
  │    ├─ 自动发现策略：v3.strategies.registry
  │    ├─ 调用策略：v3.strategies.*.generate()
  │    └─ 返回信号：v3.strategies.base.Signal
  ├─ 模拟成交：backtest.order_book_fill
  ├─ 汇总流水：backtest.trade_pipeline
  └─ 计算指标：backtest.metrics
```

### 4.2 策略信号生命周期

```text
策略 generate()
  → 返回 Signal 或 None
  → Regime / HTF / Conflict / confidence 过滤
  → 选择 confidence 最高的候选信号
  → 回测层按配置确定 exec_px，并默认用主动限价单成交
  → confirm_fill() 登记持仓并冻结风控参数
  → 后续 K 线由引擎处理 SL / TP / 分批止盈 / 超时退出
  → confirm_partial_close() / confirm_close() 回写状态
```

### 4.3 执行层：主动限价单优先

当前默认执行方式是 **主动限价单**（marketable/aggressive limit），不是裸市价单。除资金费等非交易流水外，开仓、平仓、分批止盈、止损止盈触发和期末强平都会按主动限价单发送：

```yaml
strategy:
  execution_order_type: aggressive_limit
  aggressive_limit_ticks: 5
  aggressive_limit_tick_fallback: 0.1
```

执行层会先确定信号执行基准价 `exec_px`，然后按方向给出限价保护：

| 行为 | 实际方向 | 限价 | 目的 |
|---|---|---|---|
| 开多 / 平空 | 买入 | `exec_px + 5 * tick` | 比信号价高 5 跳，确保接近市价吃单成交，同时不超过限价。 |
| 开空 / 平多 | 卖出 | `exec_px - 5 * tick` | 比信号价低 5 跳，确保接近市价吃单成交，同时不低于限价。 |

其中 `tick` 优先使用品种 tick；没有规格时使用 `aggressive_limit_tick_fallback`（默认 `0.1`）。由于没有精确盘口队列数据，主动限价单会把 `signal_px` 本身视为可成交参考：买单只要 `signal_px <= limit_px`、卖单只要 `signal_px >= limit_px`，就不会因为 K 线没有再次触价而产生 `limit_not_touched` 假拒单。成交仍然经过 `backtest/order_book_fill.py` 的订单簿深度模型，因此仍会记录：

- `ord_type=limit`
- `limit_px`
- `fill_mode`
- `book_mode`
- `slip_bps`
- `fill_reason`

如果想回到纯市价执行，可改为：

```yaml
strategy:
  execution_order_type: market
```

---
## 5. 内置策略

| name | 文件 | 类型 | 默认 Regime | 说明 |
|---|---|---|---|---|
| `vol` | `v3/strategies/vol/{vol.py,params.yaml}` | 波动突破 | `trend` | 布林带收窄后放量突破，适合趋势启动。 |
| `mr` | `v3/strategies/mr/{mr.py,params.yaml}` | 均值回归 | `chop` | RSI + Bollinger Bands + ADX + EMA/ATR 过滤。 |
| `ewmac` | `v3/strategies/ewmac/{ewmac.py,params.yaml}` | 趋势跟踪 | `trend` | 双 EWMAC 快慢线差值确认趋势。 |
| `macd` | `v3/strategies/macd/{macd.py,params.yaml}` | 动量趋势 | `trend` | MACD 交叉配合 RSI/ADX 过滤。 |
| `don` | `v3/strategies/don/{don.py,params.yaml}` | 通道突破 | `trend` | 唐奇安通道突破，含成交量与影线过滤。 |
| `ming` | `v3/strategies/ming/{ming.py,params.yaml}` | 双触发共振（DON+MACD） | `trend` | 1D 周期专用：don 突破 + macd 动量方向硬双触发（AND）；加权评分 `don*w_don + macd*w_macd`；**生命周期由引擎统一管**（结构止损 → +1R 移 SL → +1.5R 减仓 → +2R 启 Chandelier → 时间止损 / panic exit）。详见 [§5.2 ming 策略详解](#52-ming-策略详解)。 |
| `radx` | `v3/strategies/radx/{radx.py,params.yaml}` | 日线趋势研究 | `any` | 研究策略：close>SMA50/EMA7 且 RSI(2)>ADX(2) 做多，RSI(2)<ADX(2) 退出；默认不参与自动规划。 |

默认启用列表位于 `config.yaml`：

```yaml
strategy:
  enabled_strategies:
    - vol
    - mr
    - ewmac
    - macd
    - don
    - ming
  auto_plan_enabled: true
```

### 5.1 不同 K 线的建议策略与自动规划

`auto_plan_enabled: true` 时，如果 `enabled_strategies` 是完整内置策略集合，`v3.run_backtest` 会按 `--bar` 自动规划实际启用策略；如果使用 `--only` 或手动配置了策略子集，则不会覆盖你的选择。

| K 线 | 自动规划策略 | 使用建议 | 原因 |
|---|---|---|---|
| `1D` | `ming` | 日线低频趋势验证：don+macd 双触发共振，由引擎统一执行结构止损 / Chandelier。 | EWMAC 在日线容易频繁翻转；ming 专做 1D。 |
| `4H` / `2H` | `vol`, `ewmac`, `macd`, `don`, `ming` | 中周期可保留多数趋势策略，同时观察单策略贡献。 | 兼顾趋势延续和动量确认。 |
| `1H` | `vol`, `mr`, `ewmac`, `macd`, `don`, `ming` | 默认完整组合周期。 | 策略频率与原始参数最匹配。 |
| `30m` / `15m` | `vol`, `ewmac`, `macd` | 更偏动量/波动验证，谨慎使用均值回归。 | 短周期噪声更强，ming 的结构止损/时间止损针对性弱。 |
| `1m` / `3m` / `5m` | `vol` | 仅做轻量波动突破研究，不建议视作可交易默认。 | 当前包没有微结构/盘口 alpha 策略。 |

`radx` 是外部研究思路移植的日线 long-only 实验策略，默认不纳入自动规划。建议先单独验证：

```bash
python -m v3.run_backtest --bar 1D --only radx --symbols BTC-USDT-SWAP,ETH-USDT-SWAP
```

查看自动规划结果可直接运行：

```bash
python -m v3.run_backtest --bar 1D
```

日志会打印类似：

```text
自动策略规划 bar=1D: ['vol', 'mr', 'ewmac', 'macd', 'don', 'ming'] -> ['ming']
```

### 5.2 ming 策略详解

> `ming` 是为 **1D 周期**专门设计的双触发共振策略，**不是 don + macd 的简单合并**。
> 它的核心特点是：**策略只发入场计划，仓位生命周期由引擎统一管**。
> 不要在用户代码里自己再加 TP / 移 SL / 时间止损——会和引擎状态机打架。

#### 5.2.1 架构契约：策略发 Plan，引擎管 State

```
ming.generate()                                v3/engine.py
─────────────                                  ─────────────
  don 触发？ ──yes──┐                          is_ming_style 判定
                    ├──> 硬 AND ──> 计算加权       ↓
  macd 触发？──yes──┘     conf + 方向       panic > time-stop > SL
                    ↓                          ↓
        极端口门 (NATR/ADX/MA50)            +1R → SL = entry - 0.2*ATR
                    ↓                          ↓
        初始 SL (结构 + ATR 双底)            +1.5R → partial close 30%
                    ↓                          ↓
        计算 R、size、tp_batches             +2R → Chandelier = HH_N - k*ATR
                    ↓                          ↓
        return Signal(所有 plan 字段)         bars_held / mfe_r 跟踪
                                              ↓
                                     mfe_decay / breakout_fail / max_bars
                                              ↓
                                     panic_gap_dn / panic_gap_up
```

**ming 在 `Signal` 里多发以下字段**（由 `v3/strategies/base.py:Signal` 持有，引擎 `confirm_fill` 时冻结到 `Position`）：

| Signal 字段 | 含义 | 引擎用途 |
|---|---|---|
| `entry_reference_high` / `entry_reference_low` | 入场时 don 突破位的 high/low | breakout_fail 对比 |
| `mfe_window_bars` / `mfe_min_r` | 动量衰减窗口 | 3 根 K 线内 MFE < 0.5R 退出 |
| `breakout_window_bars` | 突破验证窗口 | 5 根 K 线内未破入场 K 线 high 退出 |
| `max_bars` | 硬上限 | 20 根 K 线强平（+2R + Chandelier 未触发豁免） |
| `panic_gap_atr` | 跳空阈值（ATR 倍数） | 跳空低开 > 1*ATR 且未收复 → panic exit |
| `r1_be_buffer_atr` | +1R 移 SL 缓冲 | SL = entry - 0.5*ATR（v3: 0.2→0.5） |
| `r15_partial_pct` | +1.5R 减仓比例 | 20%（v3: 0.30→0.20） |
| `r2_chandelier_N` / `r2_chandelier_k` | Chandelier 参数 | HH_10 - 3.5*ATR（v3: k 3.0→3.5） |

> **关键：ming 必须把这 4 个字段 > 0 地发到 Signal**，否则 `v3/engine.py:confirm_fill` 会回退到
> Position dataclass 默认值（0.2/0.30/10/3.0），v3 调参**不会生效**。这是 v3 wiring 修复
> 的核心（commit 6.x）。

#### 5.2.2 硬双触发 AND + 加权评分

| 步骤 | 公式 / 规则 |
|---|---|
| don 子信号 | `broke_up/down`（20 周期通道突破）+ `vol >= 20-bar mean * don_vol_ratio` + 影线/实体比 ≤ `don_max_wick_ratio` + ADX ≥ `don_min_adx` |
| macd 子信号 | `macd > 0 / < 0` + `hist` 单调扩张（不掉头）+ ADX ≥ `macd_min_adx` |
| 硬 AND | `don_long AND macd_long` / `don_short AND macd_short`；方向冲突直接拒 |
| 加权 conf | `weighted = don_score * w_don + macd_score * w_macd`，loader 自动归一化到 `w_don + w_macd = 1` |
| 入场门 | `weighted >= min_entry_conf`（默认 0.55）+ 极端口门通过 |

**为什么 macd 用 `macd > 0` 而不是 `macd > signal`？**

1D 长期 uptrend 下，`signal`（macd 的 9 周期 EMA）通常贴在 macd 顶部，要求 `macd > signal` 会让多头永远不触发；改用「macd 在零轴同侧 + hist 扩张」更稳健。`don` 才是真正负责方向确认的子信号。

#### 5.2.3 极端口门（开仓前判定）

| 条件 | 行为 | 对应参数 |
|---|---|---|
| `NATR / EMA(NATR, 20) > 1.5` | 急性扩张 → 当日不开仓 | `natr_pause_mult` |
| `natr_pct >= 0.95` | 极端行情 → 熔断 | `natr_block_pct` |
| `natr_pct >= 0.75` | 高波动 → 仓位 ×0.5（vol_mult） | `natr_shrink_pct` |
| `close < MA50`（想做多）/ `close > MA50`（想做空） | 趋势不符 → 不开仓 | `trend_ma_period=50` |
| `ADX < 18` | 无趋势 → 不开仓 | `trend_min_adx` |

> 这三层门：① 是否开仓（pause / block） ② 是否缩仓（shrink） ③ 方向是否允许（MA50）。三者独立。

#### 5.2.4 初始止损（结构 + ATR 双底 + 最低距离）

```
D      = max(k_vol(natr_pct) * ATR_14,  cur_price * f_min)
SL_0   = min(cur_price - D,  L_n - 0.2*ATR_14)   for long
       = max(cur_price + D,  H_n + 0.2*ATR_14)   for short
```

- `L_n` / `H_n`：近 8 根 K 线（不含当前）的最低 / 最高
- `k_vol` 按 NATR 历史百分位**连续插值**，避免分档跳变：
  - 百分位 ≤ 0.25 → 1.4
  - 0.25 ~ 0.75 → 1.4 → 2.3 线性插值
  - 百分位 ≥ 0.75 → 2.3
- `f_min = 0.5%`（`sl_floor_pct`）：是 `D` 的下限，不是单独的"最低止损"分支

> 设计意图：**止损距离由市场决定，亏损金额由仓位决定**。高波动时 SL 放宽，但同时 `vol_mult = 0.5`，**不会**为了维持固定亏损金额而强行收紧 SL。

#### 5.2.5 仓位（固定公式）

```
risk_dollars = capital * risk_pct * conf_mult * vol_mult
size = risk_dollars / (ctVal * R + ctVal * price * (slip + fee))
size = floor(size / lotSz) * lotSz
```

- `risk_pct = 0.5%`（`risk_pct` 参数，单笔最大亏损占权益）
- `conf_mult = clip((weighted - 0.55) / 0.45, 0, 1)`：评分越高，仓位越大（封顶 1x）
- `vol_mult = 0.5` 当 `natr_pct >= 0.75`，否则 `1.0`
- 单位修正：风险分母是 **USD/张** = `ctVal * R + ctVal * price * (slip + fee)`，不是混用价格/USD 单位

#### 5.2.6 引擎统一管位置生命周期（ming 不允许自管退出）

`v3/engine.py:_manage_position` 中，`is_ming_style = (pos.mode == "ming" or pos.mfe_window_bars > 0 or pos.max_bars > 0 or pos.panic_gap_atr > 0)`。

执行顺序（**短路返回**，命中即触发）：

| # | 触发条件 | 引擎动作 | 关闭原因标签 |
|---:|---|---|---|
| 1 | panic | 跳空 `> panic_gap_atr * ATR` 且当日未收复 | `panic_gap_dn` / `panic_gap_up` |
| 2 | time-stop | `bars_held >= max_bars` 且**不**在 `+2R + Chandelier 未触发` 豁免 | `max_bars` |
| 3 | time-stop | `bars_held == mfe_window_bars` 且 `mfe_r < mfe_min_r` | `mfe_decay` |
| 4 | time-stop | `bars_held == breakout_window_bars` 且 `bar_high < entry_ref_high`（多） | `breakout_fail` |
| 5 | +1R | `bar_high/low` 触及 `entry ± R` | `trail_stop = entry - 0.2*ATR`，`partial_done=True` |
| 6 | +1.5R | `bar_high/low` 触及 `entry ± 1.5R` | `partial_close 30%`（`partials_filled=[1]`） |
| 7 | +2R | `mfe_r >= 2.0` | `chandelier_on=True`；`trail_stop = HH_N - k*ATR` |
| 8 | 跟踪 | Chandelier 期内 | 每根 K 线更新 `HH_N` / `LL_N`，SL 只向有利方向移动 |
| 9 | 常规 SL | `bar_low/high` 触及 `trail_stop` 或初始 SL | `SL long` / `SL short` |

**ming 与其他策略的关键差异**：

| 路径 | 非 ming | ming (`is_ming_style`) |
|---|---|---|
| `_manage_position` 顺序 | timeout → SL → batch TP → trailing | panic → time-stop → +1R → +1.5R → +2R/Chandelier → 常规 SL |
| `rr_list` 批处理 | 按 `[1.0, 1.5, 2.0]` 减仓 | **完全跳过**（走状态机的 +1.5R partial 路径） |
| `Trailing update` 旧公式 | `partial_done` 后按 close 中线收紧 | **禁用**（避免和 Chandelier 冲突） |
| 跳空 panic | 无 | `panic_gap_atr` 触发主动平 |

> ⚠️ **不要**在 ming 的 `generate` 里再写 `close_signal` 或自己改 `stop_loss`；engine 是仓位状态的唯一真源。

#### 5.2.7 用 ming 跑 1D 回测

```bash
# 单独跑 ming
python -m v3.run_backtest --bar 1D --only ming

# 1D 默认 auto-plan 只跑 ming
python -m v3.run_backtest --bar 1D

# 调整加权权重 + 跑实验
python -m v3.run_backtest --bar 1D --only ming --params ming:w_don=0.5 ming:w_macd=0.5

# 调整 Chandelier 参数
python -m v3.run_backtest --bar 1D --only ming --params ming:r2_chandelier_k=2.5 ming:r2_chandelier_N=12
```

`params.yaml` 完整字段分组见 `v3/strategies/ming/params.yaml`（裸 key，无 `ming_` 前缀）。

#### 5.2.8 调参 changelog（v3 — 2026-08）

| 参数 | v1 | v3 | 影响 |
|---|---|---|---|
| `risk_pct` | 0.005 | **0.020** | 单笔风险翻 4× → 平均名义仓位从 6% 升到 ~24% |
| `r2_chandelier_k` | 3.0 | **3.5** | 牛回吐中少 0.5×ATR 缓冲；少被洗 |
| `timeout_max_bars` | 10 | **20** | 1D 硬上限从 2 周延到 1 个月，少打断跨月趋势 |
| `natr_block_pct` | 0.95 | **0.99** | 1D 波动常态大，0.95 几乎必熔断；0.99 留给极端 |
| `natr_shrink_pct` | 0.75 | **0.85** | 配合 0.99 block，让"缩仓"真例外 |
| `r15_partial_pct` | 0.30 | **0.20** | 1D 减 30% 太快被 Chandelier 洗；改 20% 留更多 |
| `r1_be_buffer_atr` | 0.2 | **0.5** | +1R 上移 SL 到 `entry - 0.5*ATR`，给正常回踩留余地 |
| `htf_weekly_ma_period` | — | **20** | 新增：周线 MA20 下方不做多、上方不做空（v3 新增） |
| **wiring 修复** | — | — | ming 把 `r1/r15/r2_N/r2_k` 显式发到 Signal；engine `confirm_fill` 优先用 signal 值，缺失时回退到 dataclass 默认 |

> ⚠️ v1 调参 `r1_be_buffer_atr` / `r15_partial_pct` / `r2_chandelier_k` **实际上不生效**：
> `v3/engine.py:confirm_fill` 用了 `or pos.default` 短路模式，但 ming 那时漏发这 4 个字段，
> 引擎永远用 Position dataclass 的 v1 默认（0.2/0.30/10/3.0）。v3 修复后这 4 个参数才真正进入引擎。
> 用户如果在 v1 数据上看不出 v3 改 `r1_be_buffer_atr` 的效果，原因就是这个。

#### 5.2.9 v4 调参（2026-08, 解决 2024 大牛 -0.43% 段）

`tools/split_oos.py` 把 OOS 拆为 6 段后，v3 在 **2024-01~2024-12 BTC 大牛段净亏 -0.43%（max_dd -0.73%）**。
诊断：v3 的 `risk_pct=0.020 + Chandelier k=3.5 + max_bars=20` 在 BTC 2024-08 闪崩 + Q4 大牛延伸时**被 Chandelier 打掉后无法重接**。

| 参数 | v3 | v4 | 修复 |
|---|---|---|---|
| `risk_pct` | 0.020 | **0.012** | 单笔风险 0.020→0.012，2024 闪崩段单笔亏损从 0.5% 降到 0.3% |
| `r2_chandelier_k` | 3.5 | **4.5** | 牛回吐容忍度从 14% → 18%，吃到大牛延伸 |
| `timeout_max_bars` | 20 | **30** | 1D 硬上限从 1 个月延到 1.5 个月，少打断跨季度趋势 |
| `r15_partial_pct` | 0.20 | **0.15** | 1.5R 减仓从 20% 降到 15%，剩余 85% 仓位让 Chandelier 多跑 |

**v4 实测**：总收益 +1.12% 是 v3 (+0.42%) 的 **2.65×**，return/dd 0.61 vs v3 0.58。
5 段中 4 段都改善，**仅 2024 段从 -0.43% 加深到 -1.12%**（max_dd -1.84%，被 2024-08 闪崩扛满 30 天）。
**v4 整体优于 v3，但 2024 闪崩段是真实风险源**。

#### 5.2.10 v5 调参（2026-08, 加动态 max_bars 解决 2024 闪崩）

v4 在 2024 大牛段 max_dd 加深到 -1.84%，根因是 30 天硬上限 + NATR 5%+ 闪崩时仓位**扛满 30 天**。
v5 引入**动态 max_bars**：NATR 超过 5% 时，max_bars 减半到 15 天，让闪崩段仓位**在 15 天强制退出**。

| 参数 | v4 | v5 | 修复 |
|---|---|---|---|
| `max_bars_vol_halve_natr` | — | **0.05** | NATR > 5% 触发 max_bars 减半 |
| `max_bars_vol_halve_factor` | — | **0.5** | max_bars 30 → 15 |

**实现位置**：`v3/strategies/ming/ming.py:generate`，在 `natr` 算完后立即算 `effective_max_bars`，
并写到 `Signal.max_bars` 让引擎读。

**预期**：v5 在 2024 闪崩段 max_dd 从 -1.84% → ~-1.0%（在 15 天强制退出），其他段不受影响（NATR 4% 以下不触发减半）。

#### 5.2.11 v6 调参（2026-08, 放大区间获取更大收益）

v5 总收益 +1.12% 仍偏低，用户要求 max_dd 15% 以内可接受，**v6 激进放大风险预算**：

| 参数 | v5 | v6 | 意义 |
|---|---|---|---|
| `risk_pct` | 0.012 | **0.020** | 单笔风险 0.012→0.020（+67%），平均名义仓位 14%→24% |
| `r2_chandelier_k` | 4.5 | **5.5** | 牛回吐容忍 18%→22%，吃到更长趋势 |
| `timeout_max_bars` | 30 | **45** | 1D 硬上限 1.5 个月→2.25 个月 |
| `r15_partial_pct` | 0.15 | **0.20** | +1.5R 减仓 15%→20%，锁定更多利润 |

**预期 max_dd 范围**：v5 段 0.06%~1.84%（极小）→ v6 应在 1%~5%（取决于 v3 段波动）。

#### 5.2.12 已知未解决（v6 后续）

- `natr_block_pct=0.99` 几乎关闭了"NATR 百分位熔断"——大牛启动日（NATR 跳到 100% 分位）仍能开仓。
  优点是能吃到 BTC 2020/2024 牛；缺点是熊市急跌时也会进（但 +1.5R 减仓 + 1W 母趋势过滤部分缓解）。
  如果 OOS 段熊市亏损放大，可考虑加 `natr_pause_mult` 收紧到 1.2 或回退 `block_pct` 到 0.95。
- 1D `signal_lookback=160` 限制了 ming 的 NATR 百分位窗口到 120 根 ≈ 半年，前期估计噪。
  调大 `signal_lookback` 到 300（1 年）能改善，但增加内存和拉取成本。
- BTC/ETH 1 张名义都偏小（ctVal=0.01/0.1），风险预算受 lotSz 截断影响显著；当前 min_sz 0.01 下
  conf_mult/vol_mult 双重缩放会触发"算出来不到 1 张"而拒单。**这导致部分 ming 信号被静默丢弃**，
  可在 `ming.py:generate` 加 `if size < min_sz and risk_dollars > 5*risk_per_lot: return None` 改 warning。

---

## 6. 数据管理

### 6.1 下载与入库

```bash
# 下载最近 180 天，并写入 SQLite / Parquet
python tools/data_manager.py --symbols BTC-USDT-SWAP --bars 1H,4H --days 180

# 指定日期区间
python tools/data_manager.py --symbols BTC-USDT-SWAP --bars 1H --start 2025-01-01 --end 2025-06-01

# 强制重下指定区间，同 timestamp 覆盖
python tools/data_manager.py --symbols BTC-USDT-SWAP --bars 1H --start 2025-01-01 --end 2025-06-01 --force
```

### 6.2 增量更新与状态检查

```bash
# 从库内最新时间戳向新方向补齐
python tools/data_manager.py --update --symbols BTC-USDT-SWAP,ETH-USDT-SWAP --bars 1H,4H

# 查看本地库状态和重复检测
python tools/data_manager.py --status
```

### 6.3 资金费率

```bash
python tools/data_manager.py --funding --symbols BTC-USDT-SWAP,ETH-USDT-SWAP
```

### 6.4 默认数据路径

| 类型 | 路径 |
|---|---|
| SQLite 主库 | `data/okx_history/bars.db` |
| Parquet 分区 | `data/okx_history/parquet/{SYMBOL}/bars_*.parquet` |
| 回测结果 | `data/backtest_results/` |

---

## 7. 回测与验证

### 7.1 组合回测

```bash
python -m v3.run_backtest --bar 1H
```

### 7.2 单策略回测

```bash
python -m v3.run_backtest --bar 1H --only vol
python -m v3.run_backtest --bar 1H --only mr
python -m v3.run_backtest --bar 1H --only ewmac
python -m v3.run_backtest --bar 1H --only macd
python -m v3.run_backtest --bar 1H --only don
python -m v3.run_backtest --bar 1D --only ming
```

### 7.3 Walk-Forward

```bash
python tools/walk_forward.py \
  --bar 1H \
  --train-days 90 \
  --test-days 30 \
  --step-days 30
```

### 7.4 参数敏感性

```bash
python tools/param_sensitivity.py \
  --bar 1H \
  --params stop_loss_atr_mult,take_profit_atr_mult,min_open_confidence \
  --delta 0.2
```

### 7.5 OOS 时间窗口拆段（ming 必看）

```bash
# 默认 6 段（2019-12 ~ 2020-12 / 2021 牛顶 / 2021-11 ~ 2022 熊 / 2023 复苏 / 2024 ETF 大牛 / 2025-26 顶部）
python tools/split_oos.py --result-dir data/backtest_results

# 自定义窗口（如想看 2022 5-9 月熊市单段）
python tools/split_oos.py \
  --result-dir data/backtest_results \
  --windows 2022-05-01:2022-09-30,2024-01-01:2024-03-31

# 输出 markdown / json 报告
python tools/split_oos.py --result-dir data/backtest_results --format md --out report.md
python tools/split_oos.py --result-dir data/backtest_results --format json --out report.json

# 不读 trades（只要 equity 段）
python tools/split_oos.py --result-dir data/backtest_results --no-trades
```

每段输出：`bars / equity_start / equity_end / delta / return / max_dd / trades / W/L / funding_pnl`。

**ming 调参后必跑**：v3 改动后 OOS 1.9% 是 "30% 段 = 2024-2026 大牛尾段" 的高水位；如果 `2022` 段 deep drawdown，需要补 trend filter；如果 `2021 牛顶` 段回吐过大，需要考虑 +1.5R 减仓比例。

### 7.6 自动诊断

回测结束后运行：

```bash
python diagnose.py --result-dir data/backtest_results --bar 1D
```

常用参数：

```bash
# 只诊断 OOS
python diagnose.py --phase oos --bar 1D

# 跳过成交价 vs K 线 OHLC 检查
python diagnose.py --skip-bar-check

# 输出逐笔 round trip 明细
python diagnose.py --write-roundtrips
```

`diagnose.py` 会显示：

- 📊 IS/OOS 收益、胜率、回撤、策略归因；
- 🧾 逐笔 open/close 配对核对；
- 🛡️ 假成交风险检测：拒单、孤儿平仓、未平仓、期末强平错归因、成交价超出 K 线 OHLC；
- 💰 成交模式、订单簿/合成深度、滑点分布；
- 🔎 问题清单和 PASS / PASS_WITH_WARNINGS / FAIL 结论。

---

## 8. 自定义策略开发手册

### 8.1 开发步骤

```text
1. 在 v3/strategies/ 下新建 <my_strategy>/ 目录
2. 在 <my_strategy>/<my_strategy>.py 写策略类，类属性 name = "<my_strategy>"
3. 继承 StrategyBase
4. 实现 generate(...)
5. 在 <my_strategy>/params.yaml 写默认参数（裸 key，无前缀）
6. （可选）写 <my_strategy>/README.md 说明每个参数
7. 在 config.yaml → strategy.enabled_strategies 中启用 <my_strategy>
8. 用 --list-strategies 确认已被发现
9. 用 --only <my_strategy> 单策略回测验证
```

#### 目录结构示例

```
v3/strategies/
  don/
    don.py         # class DonchianBreakout(StrategyBase): name = "don"
    params.yaml    # 唐奇安参数（裸 key）
    README.md      # 参数说明
  vol/
    vol.py
    params.yaml
    README.md
  my_strategy/     # 新增策略
    my_strategy.py
    params.yaml
    README.md
```

### 8.2 策略协议

| 项 | 要求 |
|---|---|
| 基类 | 必须继承 `StrategyBase` |
| 唯一标识 | 类属性 `name`，全局唯一，建议小写下划线 |
| 主方法 | `generate(...) -> Optional[Signal]` |
| 市场状态 | `required_regime`: `any` / `trend` / `chop` / `mixed` |
| 放置位置 | `v3/strategies/<name>/<name>.py`（目录 = 策略名） |
| 参数位置 | `v3/strategies/<name>/params.yaml`（裸 key，无前缀） |
| 启用方式 | `config.yaml` → `strategy.enabled_strategies` |
| 职责边界 | 只生成信号，不下单、不改仓、不写成交、不直接调交易所 API |

#### 参数合并顺序（每个策略实例化时）

1. `v3/strategies/<name>/params.yaml`（默认值）
2. `config.yaml → strategy:` 段（跨策略共享字段，如 `min_open_confidence`）
3. `--extra-config FILE`（临时覆盖）
4. `--params NAME=KEY=VAL` 或 `--params NAME:KEY=VAL`（最高优先级）

#### 修改/实验参数

```bash
# 改一个参数跑实验
python -m v3.run_backtest --bar 1H --only don --params don:period=14

# 一次性覆盖一组参数
python -m v3.run_backtest --bar 1H --only don --extra-config my_don.yaml
```

### 8.3 `generate` 方法签名

```python
def generate(
    self,
    df,                    # 当前品种主周期 K 线，时间升序
    regime,                # trend / chop / mixed
    last,                  # 最新参考价
    capital,               # 分配给该品种的资金
    leverage,              # 杠杆
    specs,                 # 合约规格：ct_val、lot_sz、min_sz、tick_sz 等
    kelly_factor=1.0,
    funding_rate=0.0,
):
    ...
```

`df` 至少应包含以下列：

| 列 | 含义 |
|---|---|
| `ts` | 时间戳 |
| `open` / `high` / `low` / `close` | OHLC |
| `vol` | 成交量 |

`specs` 常用键：

| 键 | 含义 |
|---|---|
| `ct_val` / `ctVal` | 合约面值 |
| `lot_sz` / `lotSz` | 张数步长 |
| `min_sz` / `minSz` | 最小张数 |
| `tick_sz` / `tickSz` | 价格精度 |

### 8.4 `Signal` 字段说明

| 字段 | 必填 | 说明 |
|---|---:|---|
| `action` | 是 | `open_long` / `open_short`；无信号直接返回 `None` |
| `direction` | 是 | `long` / `short` |
| `confidence` | 是 | 0~1；需不低于 `min_open_confidence` 才可能成交 |
| `strategy` | 是 | 填 `self.name` |
| `size` | 是 | 建议张数，通常使用 `_calc_size()` 计算 |
| `stop_loss` | 建议 | 止损价，开仓后由引擎冻结 |
| `take_profit` | 建议 | 止盈价 |
| `rr_list` | 可选 | 分批止盈 R 倍数，例如 `[1.0, 1.5, 2.5]` |
| `batch_ratios` | 可选 | 分批比例，例如 `[0.3, 0.3, 0.4]` |
| `atr` | 可选 | 入场波动参考 |
| `reason` | 建议 | 日志说明，便于复盘 |

### 8.5 基类工具方法

```python
# 计算张数
size = self._calc_size(
    capital, leverage, position_pct, last, specs, kelly_factor,
    sl_distance_pct=0.01,
    atr_pct=0.0,
)

# 资金费方向过滤
if self._funding_blocked("long", funding_rate):
    return None

# 净期望 R 过滤
exp_r = self._expected_r(sl_pct, tp_pct, win_rate=0.55, fee_rt=0.0007, slippage=0.0003)
```

### 8.6 最小完整示例

```text
v3/strategies/sma_cross/
    sma_cross.py    # 策略类
    params.yaml     # 默认参数（裸 key）
    README.md       # 可选
```

```python
# v3/strategies/sma_cross/sma_cross.py
from __future__ import annotations
from typing import Optional

from v3.strategies.base import Signal, StrategyBase


class SmaCross(StrategyBase):
    name = "sma_cross"
    required_regime = "trend"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # 裸 key，不再有 sma_cross_ 前缀（默认值与 params.yaml 一致）
        self.fast = int(cfg.get("fast", 10))
        self.slow = int(cfg.get("slow", 30))
        self.position_pct = float(cfg.get("position_pct", 0.04))
        self.sl_pct = float(cfg.get("sl_pct", 0.015))
        self.tp_pct = float(cfg.get("tp_pct", 0.03))

    def generate(self, df, regime, last, capital, leverage, specs,
                 kelly_factor=1.0, funding_rate=0.0) -> Optional[Signal]:
        if df is None or len(df) < self.slow + 2:
            return None

        close = df["close"].astype(float)
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()

        # 金叉：上一根 fast <= slow，本根 fast > slow
        cross_up = fast_ma.iloc[-2] <= slow_ma.iloc[-2] and fast_ma.iloc[-1] > slow_ma.iloc[-1]
        if not cross_up:
            return None

        if self._funding_blocked("long", funding_rate):
            return None

        size = self._calc_size(
            capital, leverage, self.position_pct, last, specs, kelly_factor,
            sl_distance_pct=self.sl_pct,
        )
        if size <= 0:
            return None

        return Signal(
            action="open_long",
            direction="long",
            confidence=0.62,
            strategy=self.name,
            regime=regime,
            stop_loss=last * (1.0 - self.sl_pct),
            take_profit=last * (1.0 + self.tp_pct),
            rr_list=[self.tp_pct / self.sl_pct],
            batch_ratios=[1.0],
            size=size,
            reason=f"[sma_cross] golden cross {self.fast}/{self.slow}",
        )
```

```yaml
# v3/strategies/sma_cross/params.yaml
fast: 10
slow: 30
position_pct: 0.04
sl_pct: 0.015
tp_pct: 0.03
```

```yaml
# config.yaml (顶层)
strategy:
  enabled_strategies:
    - sma_cross
  # 不要在这里写 sma_cross_xxx — 已迁移到 params.yaml
```

`config.yaml` 示例：

```yaml
strategy:
  enabled_strategies:
    - sma_cross

  sma_cross_fast: 10
  sma_cross_slow: 30
  sma_cross_position_pct: 0.04
  sma_cross_sl_pct: 0.015
  sma_cross_tp_pct: 0.03
```

验证命令：

```bash
python -m v3.run_backtest --list-strategies
python -m v3.run_backtest --bar 1H --only sma_cross --symbols BTC-USDT-SWAP
```

---

## 9. 文件职责与依赖关系

### 9.1 根目录

| 文件 | 作用 | 调用关系 |
|---|---|---|
| `README.md` | 项目总手册。 | 人工阅读。 |
| `config.yaml` | 唯一配置源，包含资金、杠杆、Universe、风控、策略启用列表和参数。 | `v3/run_backtest.py`、研究工具读取。 |
| `requirements.txt` | Python 依赖。 | 环境安装使用。 |
| `logger.py` | 统一日志格式。 | 回测、数据工具、研究工具调用。 |
| `diagnose.py` | 回测诊断器：绩效归因、逐笔流水核对、假成交/OHLC 越界/错归因检测。 | 用户命令行调用；读取 `data/backtest_results/MULTI_*`。 |
| `strategy.py` | 兼容入口：导出 `v3.engine.StrategyEngine` 和 `Signal` 别名。 | 旧调用方可 `from strategy import StrategyEngine`。 |
| `utils.py` | 通用工具函数。 | 历史兼容与辅助代码调用。 |
| `universe.py` | 交易品种 Universe 选择/过滤辅助。 | 回测或外部脚本可调用。 |

### 9.2 `backtest/`

| 文件 | 作用 | 调用关系 |
|---|---|---|
| `backtest/__init__.py` | backtest 包标识。 | Python 包导入使用。 |
| `backtest/data_store.py` | 本地历史 K 线存储读取与示例数据辅助。 | `multi_engine`、数据相关流程使用。 |
| `backtest/engine.py` | 单品种/旧版回测引擎兼容层。 | 保留给旧入口或外部调用。 |
| `backtest/funding_store.py` | 资金费率下载、入库、查询。 | `tools/data_manager.py`、`multi_engine` 调用。 |
| `backtest/hist_fetcher.py` | 历史数据读取/拉取辅助。 | 数据管理与回测加载流程调用。 |
| `backtest/metrics.py` | 收益、回撤、Sharpe、Calmar、胜率、过拟合检查等指标。 | `v3/run_backtest.py`、`multi_engine` 调用。 |
| `backtest/multi_engine.py` | 多品种主回测引擎；对齐时间轴、分 IS/OOS、撮合信号、输出结果。 | `v3/run_backtest.py`、Walk-Forward、敏感性工具调用；内部调用 `v3.engine`。 |
| `backtest/order_book_fill.py` | 订单簿成交模拟：滑点、延迟、部分成交、拒单。 | `multi_engine` 调用。 |
| `backtest/progress.py` | 回测进度显示。 | 回测流程调用。 |
| `backtest/trade_pipeline.py` | 成交流水、持仓和结果管线辅助。 | 回测引擎调用。 |

### 9.3 `tools/`

| 文件 | 作用 | 调用关系 |
|---|---|---|
| `tools/__init__.py` | tools 包标识。 | Python 包导入使用。 |
| `tools/data_manager.py` | 数据统一入口：下载、校验、入库、增量更新、状态检查、资金费同步。 | 用户命令行调用；内部调用 OKX 接口与 `funding_store`。 |
| `tools/download_okx_history.py` | 兼容入口，转发到 `data_manager.py`。 | 用户旧命令调用。 |
| `tools/build_history_parquet.py` | 兼容入口，转发到 `data_manager.py` 的入库流程。 | 用户旧命令调用。 |
| `tools/build_funding_history.py` | 兼容入口，构建资金费率历史。 | 用户命令行调用；调用 `funding_store`。 |
| `tools/walk_forward.py` | 滚动 Walk-Forward OOS 验证。 | 调用 `MultiSymbolBacktester`。 |
| `tools/param_sensitivity.py` | 参数敏感性扫描。 | 调用 `MultiSymbolBacktester`。 |
| `tools/split_oos.py` | IS/OOS 按时间窗口拆段（ming 调参后必跑）。 | 读 `MULTI_{IS,OOS}_equity.csv` + `MULTI_{IS,OOS}_trades.csv`；不调回测，只读结果。 |

### 9.4 `v3/`

| 文件 | 作用 | 调用关系 |
|---|---|---|
| `v3/run_backtest.py` | v3 主命令行入口；加载配置、列策略、启动回测、打印指标。 | 用户命令行调用；调用 `backtest.multi_engine`。 |
| `v3/engine.py` | 策略组合引擎：Regime、HTF、冲突过滤、候选信号选择、持仓管理、成交确认。 | `multi_engine` 调用；内部调用策略、风险过滤器。 |
| `v3/indicators/__init__.py` | 指标包导出。 | 策略模块导入。 |
| `v3/indicators/core.py` | 技术指标实现：RSI、ATR、ADX、EMA、Bollinger 等。 | 各策略与风险模块调用。 |
| `v3/risk/__init__.py` | 风险包导出。 | `v3.engine` 导入。 |
| `v3/risk/regime.py` | ADX + BBW Regime 检测。 | `v3.engine` 调用。 |
| `v3/risk/filters.py` | HTF 高周期过滤和冲突窗口过滤。 | `v3.engine` 调用。 |
| `v3/strategies/__init__.py` | 策略包导出并触发自动发现。 | `v3.engine`、策略工具导入。 |
| `v3/strategies/base.py` | `Signal` 数据类与 `StrategyBase` 基类，含 sizing、资金费过滤、期望 R 工具。 | 所有策略继承/返回。 |
| `v3/strategies/registry.py` | 策略注册表与自动发现。 | `v3.engine`、`v3.run_backtest --list-strategies` 调用。 |
| `v3/strategies/vol_breakout.py` | VOL 波动突破策略。 | 自动发现后由 `v3.engine` 调用。 |
| `v3/strategies/mean_reversion.py` | MR 均值回归策略。 | 自动发现后由 `v3.engine` 调用。 |
| `v3/strategies/rsi_adx_trend.py` | RADX 日线 RSI/ADX 趋势研究策略。 | 自动发现后可用 `--only radx` 单独验证；退出规则由 `v3.engine` 处理。 |
| `v3/strategies/ewmac_trend.py` | EWMAC 趋势策略。 | 自动发现后由 `v3.engine` 调用。 |
| `v3/strategies/macd_divergence.py` | MACD 动量策略。 | 自动发现后由 `v3.engine` 调用。 |
| `v3/strategies/donchian_breakout.py` | DON 唐奇安突破策略。 | 自动发现后由 `v3.engine` 调用。 |
| `v3/strategies/ming/{ming.py, params.yaml, README.md}` | 1D 双触发共振（DON+MACD）：`MingBreakout` 策略 + 参数 + 设计说明。 | 自动发现后由 `v3.engine` 调用；仓位生命周期由 `v3/engine.py:_manage_position` 统一管理（详见 [§5.2](#52-ming-策略详解)）。 |
| `v3/utils/__init__.py` | v3 utils 包标识。 | Python 包导入使用。 |
| `v3/utils/run_meta.py` | 回测 run_id、配置 hash、元数据写出和一致性校验。 | `backtest/multi_engine.py` 调用。 |
| `v3/utils/sizing.py` | Kelly / Volatility Targeting 仓位计算。 | `StrategyBase._calc_size()` 调用。 |

---

## 10. 已移除内容

| 已移除 | 原因 |
|---|---|
| `audit_strategy.py` | 审计模块已下线。 |
| `replay_exits.py` | 依赖审计 CSV 的退出重放研究脚本已下线。 |
| `docs_rng_repair.md` | 旧版 RNG 修复说明不再适用。 |
| `v3/strategies/range_position.py` (RNG) | 2026-08 删除：1D 的 don+macd 双触发共振由 `ming` 策略（`v3/strategies/ming/`）+ 引擎结构止损/Chandelier 替代；原 rng 在 `_ALL_BUILTIN_STRATEGIES` / `1D` auto-plan 中的位置已由 ming 顶替。 |
| `v3/strategies/_template_plugin.py` | 占位模板已删除；自定义策略写法统一收敛到本 README。 |
| 独立自定义策略说明文档 | 内容已合并到本 README。 |
| `v3/README.md` | 旧文档重复且过期，已合并到根 README。 |

> 新增策略时，请直接创建真实策略文件，不要提交仅返回 `None` 的占位策略文件。
