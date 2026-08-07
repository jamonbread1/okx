# EWMAC — Exponential Weighted Moving Average Crossover

趋势跟踪策略，regime = `trend`。

## 核心逻辑

1. 双 EWMAC（8/32 + 16/64）共振确认
2. 快线信号绝对值 > `threshold` 才开仓
3. 慢线信号 > `threshold * 0.5` 共振（避免微小信号放行）
4. ADX > `min_adx` 才开仓

## 参数（`params.yaml`）

| key | 含义 | 默认 |
|---|---|---|
| `fast_1` | 快线 1 周期 | 8 |
| `slow_1` | 慢线 1 周期 | 32 |
| `fast_2` | 快线 2 周期 | 16 |
| `slow_2` | 慢线 2 周期 | 64 |
| `threshold` | EWMAC 信号阈值 | 0.005 |
| `min_adx` | ADX 下限 | 20.0 |
| `position_pct` | 单次开仓权益占比 | 0.05 |
| `sl_atr_mult` | 止损 ATR 倍数 | 2.0 |
| `tp_atr_mult` | 止盈 ATR 倍数 | 3.0 |
| `slippage_pct` | 单边滑点 | 0.0003 |

## 已知行为

- 实际分批止盈硬编码 `rr_list = [1.0, 2.0, 3.0]`，`tp_atr_mult` 未直接使用。
- 日线（1D）容易频繁翻转，README 建议 1D 用 macd+don 替代。
