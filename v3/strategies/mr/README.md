# MR — Mean Reversion（均值回归）

震荡区策略，regime = `chop`。

## 核心逻辑

1. 价格触碰 BB 下/上轨
2. RSI 进入超卖/超买区
3. ADX < `max_adx`（确认震荡/弱趋势）
4. EMA 趋势过滤：价格偏离 EMA 不超过 `ema_atr_mult` 个 ATR

## 参数（`params.yaml`）

| key | 含义 | 默认 |
|---|---|---|
| `bb_period` | Bollinger 周期 | 20 |
| `rsi_period` | RSI 周期 | 14 |
| `rsi_oversold` | 超卖阈值 | 30.0 |
| `rsi_overbought` | 超买阈值 | 70.0 |
| `max_adx` | ADX 上限 | 25.0 |
| `ema_filter` | EMA 周期 | 50 |
| `ema_atr_mult` | 价格偏离 EMA 的 ATR 倍数上限 | 2.5 |
| `position_pct` | 单次开仓权益占比 | 0.04 |
| `sl_atr_mult` | 止损 ATR 倍数 | 2.0 |
| `tp_atr_mult` | 止盈 ATR 倍数 | 1.5 |
| `min_expectancy` | 净期望 R 阈值 | 0.05 |
| `slippage_pct` | 单边滑点 | 0.0003 |
| `fee_rt` | 单边手续费 | 0.0007 |

## 已知行为

- 实际分批止盈是硬编码 `rr_list = [0.75, 1.5]`，`tp_atr_mult` 仅用于期望 R 计算。
- 出场由引擎层 `PositionManager` 统一处理（`exit_mode`/`position_timeout_sec`）。
