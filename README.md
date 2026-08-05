# OKX Quant 回测手册

> 面向 **OKX 永续合约历史数据** 的模块化量化回测框架。
> 当前版本专注于：数据管理、策略信号、组合回测、风险过滤、资金费率、滑点成交、Walk-Forward 与参数敏感性分析。

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img alt="Mode" src="https://img.shields.io/badge/Mode-Backtest-2E8B57?style=flat-square" />
  <img alt="Config" src="https://img.shields.io/badge/Config-config.yaml-orange?style=flat-square" />
  <img alt="Strategies" src="https://img.shields.io/badge/Strategies-VOL%20%7C%20MR%20%7C%20RNG%20%7C%20EWMAC%20%7C%20MACD%20%7C%20DON-blueviolet?style=flat-square" />
</p>

---

## 目录

- [1. 项目定位](#1-项目定位)
- [2. 快速开始](#2-快速开始)
- [3. 功能总览](#3-功能总览)
- [4. 主流程与调用关系](#4-主流程与调用关系)
- [5. 内置策略](#5-内置策略)
- [6. 数据管理](#6-数据管理)
- [7. 回测与验证](#7-回测与验证)
- [8. 自定义策略开发手册](#8-自定义策略开发手册)
- [9. 文件职责与依赖关系](#9-文件职责与依赖关系)
- [10. 已移除内容](#10-已移除内容)

---

## 1. 项目定位

本项目是一个 **纯回测发行包**，用于在 OKX 永续合约历史数据上验证多策略组合。系统当前默认策略组合为：

```text
vol + mr + rng + ewmac + macd + don
```

当前版本已下线：

- 审计 / 退出重放模块
- 旧版 RNG 区间策略（已由 RNG v4 替代）
- 仅作演示的占位模板文件
- 重复或过期的说明文档

> 策略只负责生成信号；成交、滑点、手续费、资金费、止损、止盈、超时退出全部由回测引擎处理。

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
python -m v2.run_backtest --bar 1H
```

### 2.4 常用命令

```bash
# 指定品种
python -m v2.run_backtest --bar 1H --symbols BTC-USDT-SWAP,ETH-USDT-SWAP

# 只跑一个策略
python -m v2.run_backtest --bar 1H --only rng --symbols BTC-USDT-SWAP

# 指定时间范围
python -m v2.run_backtest --bar 1H --start 2025-01-01 --end 2025-06-01

# 查看自动发现到的策略
python -m v2.run_backtest --list-strategies
```

默认结果目录：

```text
data/backtest_results_v2/
```

---

## 3. 功能总览

| 模块 | 能力 | 入口文件 |
|---|---|---|
| 统一配置 | 根目录 `config.yaml` 作为唯一配置源 | `config.yaml` |
| 数据管理 | 下载、校验、入库、增量更新、状态检查 | `tools/data_manager.py` |
| 多品种回测 | 时间轴对齐、IS/OOS 切分、组合绩效输出 | `backtest/multi_engine.py` |
| 策略引擎 | 策略发现、信号生成、候选信号选择、持仓管理 | `v2/engine.py` |
| 风险过滤 | Regime、HTF 方向过滤、冲突窗口过滤 | `v2/risk/` |
| 成交模拟 | 订单簿滑点、延迟、部分成交、拒单、手续费 | `backtest/order_book_fill.py` |
| 资金费率 | 历史资金费率入库与回测计入 | `backtest/funding_store.py` |
| 稳定性验证 | Walk-Forward、参数敏感性、过拟合检查 | `tools/walk_forward.py` / `tools/param_sensitivity.py` |
| 自动诊断 | 绩效归因、逐笔流水核对、假成交/错归因/OHLC 越界检测 | `diagnose.py` |

---

## 4. 主流程与调用关系

### 4.1 总体链路

```text
config.yaml
  ↓
v2.run_backtest.load_v2_config()
  ↓
backtest.multi_engine.run_multi_from_parquet()
  ├─ 读取历史 K 线：backtest.data_store / backtest.hist_fetcher
  ├─ 读取资金费率：backtest.funding_store
  ├─ 生成信号与管理持仓：v2.engine.StrategyEngine
  │    ├─ 判断市场状态：v2.risk.regime.RegimeDetector
  │    ├─ 过滤方向/冲突：v2.risk.filters.HtfFilter / ConflictFilter
  │    ├─ 自动发现策略：v2.strategies.registry
  │    ├─ 调用策略：v2.strategies.*.generate()
  │    └─ 返回信号：v2.strategies.base.Signal
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
| `vol` | `v2/strategies/vol_breakout.py` | 波动突破 | `trend` | 布林带收窄后放量突破，适合趋势启动。 |
| `mr` | `v2/strategies/mean_reversion.py` | 均值回归 | `chop` | RSI + Bollinger Bands + ADX + EMA/ATR 过滤。 |
| `rng` | `v2/strategies/range_position.py` | 假突破回归 | `any` | RNG v4：区间形成后，前一根假突破收回，当前根继续向区间内部确认后入场。 |
| `ewmac` | `v2/strategies/ewmac_trend.py` | 趋势跟踪 | `trend` | 双 EWMAC 快慢线差值确认趋势。 |
| `macd` | `v2/strategies/macd_divergence.py` | 动量趋势 | `trend` | MACD 交叉配合 RSI/ADX 过滤。 |
| `don` | `v2/strategies/donchian_breakout.py` | 通道突破 | `trend` | 唐奇安通道突破，含成交量与影线过滤。 |
| `radx` | `v2/strategies/rsi_adx_trend.py` | 日线趋势研究 | `any` | 研究策略：close>SMA50/EMA7 且 RSI(2)>ADX(2) 做多，RSI(2)<ADX(2) 退出；默认不参与自动规划。 |

默认启用列表位于 `config.yaml`：

```yaml
strategy:
  enabled_strategies:
    - vol
    - mr
    - rng
    - ewmac
    - macd
    - don
  auto_plan_enabled: true
```

### 5.1 不同 K 线的建议策略与自动规划

`auto_plan_enabled: true` 时，如果 `enabled_strategies` 是完整内置策略集合，`v2.run_backtest` 会按 `--bar` 自动规划实际启用策略；如果使用 `--only` 或手动配置了策略子集，则不会覆盖你的选择。

| K 线 | 自动规划策略 | 使用建议 | 原因 |
|---|---|---|---|
| `1D` | `rng`, `macd`, `don` | 日线先验证结构型假突破与低频趋势，不让 EWMAC 主导组合。 | 日线 EWMAC 容易频繁翻转；RNG 样本少时需用诊断器确认样本量。 |
| `4H` / `2H` | `vol`, `rng`, `ewmac`, `macd`, `don` | 中周期可保留多数趋势策略，同时观察单策略贡献。 | 兼顾趋势延续和假突破回归。 |
| `1H` | `vol`, `mr`, `rng`, `ewmac`, `macd`, `don` | 默认完整组合周期。 | 策略频率与原始参数最匹配。 |
| `30m` / `15m` | `vol`, `ewmac`, `macd` | 更偏动量/波动验证，谨慎使用均值回归。 | 短周期噪声更强，RNG 的区间结构更容易失真。 |
| `1m` / `3m` / `5m` | `vol` | 仅做轻量波动突破研究，不建议视作可交易默认。 | 当前包没有微结构/盘口 alpha 策略。 |

`radx` 是外部研究思路移植的日线 long-only 实验策略，默认不纳入自动规划。建议先单独验证：

```bash
python -m v2.run_backtest --bar 1D --only radx --symbols BTC-USDT-SWAP,ETH-USDT-SWAP
```

查看自动规划结果可直接运行：

```bash
python -m v2.run_backtest --bar 1D
```

日志会打印类似：

```text
自动策略规划 bar=1D: ['vol', 'mr', 'rng', 'ewmac', 'macd', 'don'] -> ['rng', 'macd', 'don']
```

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
| 回测结果 | `data/backtest_results_v2/` |

---

## 7. 回测与验证

### 7.1 组合回测

```bash
python -m v2.run_backtest --bar 1H
```

### 7.2 单策略回测

```bash
python -m v2.run_backtest --bar 1H --only vol
python -m v2.run_backtest --bar 1H --only mr
python -m v2.run_backtest --bar 1H --only rng
python -m v2.run_backtest --bar 1H --only ewmac
python -m v2.run_backtest --bar 1H --only macd
python -m v2.run_backtest --bar 1H --only don
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

### 7.5 自动诊断

回测结束后运行：

```bash
python diagnose.py --result-dir data/backtest_results_v2 --bar 1D
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
1. 在 v2/strategies/ 下新建 my_alpha.py
2. 继承 StrategyBase
3. 设置唯一 name
4. 实现 generate(...)
5. 在 config.yaml → strategy.enabled_strategies 中启用 name
6. 用 --list-strategies 确认已被发现
7. 用 --only name 单策略回测验证
```

### 8.2 策略协议

| 项 | 要求 |
|---|---|
| 基类 | 必须继承 `StrategyBase` |
| 唯一标识 | 类属性 `name`，全局唯一，建议小写下划线 |
| 主方法 | `generate(...) -> Optional[Signal]` |
| 市场状态 | `required_regime`: `any` / `trend` / `chop` / `mixed` |
| 放置位置 | `v2/strategies/*.py`，文件名不要以 `_` 开头 |
| 启用方式 | `config.yaml` → `strategy.enabled_strategies` |
| 职责边界 | 只生成信号，不下单、不改仓、不写成交、不直接调交易所 API |

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

```python
# v2/strategies/sma_cross.py
from __future__ import annotations
from typing import Optional

from v2.strategies.base import Signal, StrategyBase


class SmaCross(StrategyBase):
    name = "sma_cross"
    required_regime = "trend"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.fast = int(cfg.get("sma_cross_fast", 10))
        self.slow = int(cfg.get("sma_cross_slow", 30))
        self.position_pct = float(cfg.get("sma_cross_position_pct", 0.04))
        self.sl_pct = float(cfg.get("sma_cross_sl_pct", 0.015))
        self.tp_pct = float(cfg.get("sma_cross_tp_pct", 0.03))

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
            size=float(size),
            reason=f"[sma_cross] golden cross {self.fast}/{self.slow}",
        )
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
python -m v2.run_backtest --list-strategies
python -m v2.run_backtest --bar 1H --only sma_cross --symbols BTC-USDT-SWAP
```

---

## 9. 文件职责与依赖关系

### 9.1 根目录

| 文件 | 作用 | 调用关系 |
|---|---|---|
| `README.md` | 项目总手册。 | 人工阅读。 |
| `config.yaml` | 唯一配置源，包含资金、杠杆、Universe、风控、策略启用列表和参数。 | `v2/run_backtest.py`、研究工具读取。 |
| `requirements.txt` | Python 依赖。 | 环境安装使用。 |
| `logger.py` | 统一日志格式。 | 回测、数据工具、研究工具调用。 |
| `diagnose.py` | 回测诊断器：绩效归因、逐笔流水核对、假成交/OHLC 越界/错归因检测。 | 用户命令行调用；读取 `data/backtest_results_v2/MULTI_*`。 |
| `strategy.py` | 兼容入口：导出 `v2.engine.StrategyEngine` 和 `Signal` 别名。 | 旧调用方可 `from strategy import StrategyEngine`。 |
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
| `backtest/metrics.py` | 收益、回撤、Sharpe、Calmar、胜率、过拟合检查等指标。 | `v2/run_backtest.py`、`multi_engine` 调用。 |
| `backtest/multi_engine.py` | 多品种主回测引擎；对齐时间轴、分 IS/OOS、撮合信号、输出结果。 | `v2/run_backtest.py`、Walk-Forward、敏感性工具调用；内部调用 `v2.engine`。 |
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

### 9.4 `v2/`

| 文件 | 作用 | 调用关系 |
|---|---|---|
| `v2/run_backtest.py` | v2 主命令行入口；加载配置、列策略、启动回测、打印指标。 | 用户命令行调用；调用 `backtest.multi_engine`。 |
| `v2/engine.py` | 策略组合引擎：Regime、HTF、冲突过滤、候选信号选择、持仓管理、成交确认。 | `multi_engine` 调用；内部调用策略、风险过滤器。 |
| `v2/indicators/__init__.py` | 指标包导出。 | 策略模块导入。 |
| `v2/indicators/core.py` | 技术指标实现：RSI、ATR、ADX、EMA、Bollinger 等。 | 各策略与风险模块调用。 |
| `v2/risk/__init__.py` | 风险包导出。 | `v2.engine` 导入。 |
| `v2/risk/regime.py` | ADX + BBW Regime 检测。 | `v2.engine` 调用。 |
| `v2/risk/filters.py` | HTF 高周期过滤和冲突窗口过滤。 | `v2.engine` 调用。 |
| `v2/strategies/__init__.py` | 策略包导出并触发自动发现。 | `v2.engine`、策略工具导入。 |
| `v2/strategies/base.py` | `Signal` 数据类与 `StrategyBase` 基类，含 sizing、资金费过滤、期望 R 工具。 | 所有策略继承/返回。 |
| `v2/strategies/registry.py` | 策略注册表与自动发现。 | `v2.engine`、`v2.run_backtest --list-strategies` 调用。 |
| `v2/strategies/vol_breakout.py` | VOL 波动突破策略。 | 自动发现后由 `v2.engine` 调用。 |
| `v2/strategies/mean_reversion.py` | MR 均值回归策略。 | 自动发现后由 `v2.engine` 调用。 |
| `v2/strategies/range_position.py` | RNG v4 假突破回归策略。 | 自动发现后由 `v2.engine` 调用；可用 `rng_timeout_bars` 设置独立持仓超时。 |
| `v2/strategies/rsi_adx_trend.py` | RADX 日线 RSI/ADX 趋势研究策略。 | 自动发现后可用 `--only radx` 单独验证；退出规则由 `v2.engine` 处理。 |
| `v2/strategies/ewmac_trend.py` | EWMAC 趋势策略。 | 自动发现后由 `v2.engine` 调用。 |
| `v2/strategies/macd_divergence.py` | MACD 动量策略。 | 自动发现后由 `v2.engine` 调用。 |
| `v2/strategies/donchian_breakout.py` | DON 唐奇安突破策略。 | 自动发现后由 `v2.engine` 调用。 |
| `v2/utils/__init__.py` | v2 utils 包标识。 | Python 包导入使用。 |
| `v2/utils/run_meta.py` | 回测 run_id、配置 hash、元数据写出和一致性校验。 | `backtest/multi_engine.py` 调用。 |
| `v2/utils/sizing.py` | Kelly / Volatility Targeting 仓位计算。 | `StrategyBase._calc_size()` 调用。 |

---

## 10. 已移除内容

| 已移除 | 原因 |
|---|---|
| `audit_strategy.py` | 审计模块已下线。 |
| `replay_exits.py` | 依赖审计 CSV 的退出重放研究脚本已下线。 |
| `docs_rng_repair.md` | 旧版 RNG 修复说明不再适用。 |
| `v2/strategies/_template_plugin.py` | 占位模板已删除；自定义策略写法统一收敛到本 README。 |
| 独立自定义策略说明文档 | 内容已合并到本 README。 |
| `v2/README.md` | 旧文档重复且过期，已合并到根 README。 |

> 新增策略时，请直接创建真实策略文件，不要提交仅返回 `None` 的占位策略文件。
