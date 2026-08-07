# VOL — Bollinger Bands Squeeze Breakout

波动率突破策略，regime = `any`。

## 核心逻辑

1. **BBW squeeze** 检测：前 `squeeze_hold` 根 BBW < `bbw_squeeze` 视为盘整末端
2. **放量确认**：`vol_ratio` >= `volume_mult`
3. **价格突破** BB 上/下轨（带 `breakout_buffer` 余量）
4. **RSI 中性区** 过滤：`rsi_low <= RSI <= rsi_high`
5. **净期望 R 过滤**：`exp_r >= min_expectancy`

## 参数（`params.yaml`）

| key | 含义 | 默认 |
|---|---|---|
| `bb_period` | Bollinger 周期 | 20 |
| `bbw_squeeze` | squeeze 阈值 | 0.035 |
| `volume_mult` | 放量倍数 | 1.3 |
| `rsi_low` | RSI 中性区下限 | 35.0 |
| `rsi_high` | RSI 中性区上限 | 65.0 |
| `breakout_buffer` | 突破缓冲 | 0.0003 |
| `position_pct` | 单次开仓权益占比 | 0.05 |
| `sl_atr_mult` | 止损 ATR 倍数 | 1.8 |
| `tp_atr_mult` | 止盈 ATR 倍数（仅用于期望 R 计算） | 3.5 |
| `min_expectancy` | 净期望 R 阈值 | 0.1 |
| `slippage_pct` | 单边滑点 | 0.00035 |
| `fee_rt` | 单边手续费 | 0.0007 |
| `short_penalty` | 做空仓位乘数 | 0.7 |
| `squeeze_hold` | squeeze 持续根数 | 3 |

## 已知行为

- 做空需要 `vol_ratio >= volume_mult + 0.3`，比做多要求更高（量价配合）。
- 实际分批止盈是硬编码的 `rr_list = [1.0, 1.5, 2.5]`，不受 `tp_atr_mult` 影响。
- HTF 趋势过滤由引擎层 `HtfFilter` 统一处理，策略内不重复。
