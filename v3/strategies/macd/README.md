# MACD — MACD 交叉 + RSI + ADX

动量趋势策略，regime = `trend`。

## 核心逻辑

1. MACD line 上穿/下穿 signal line
2. RSI 确认（命名延续旧版）：
   - 做多：`RSI < rsi_low`（默认 65）— 避免多头钝化区追涨
   - 做空：`RSI > rsi_high`（默认 35）— 避免空头钝化区追跌
3. Histogram 扩张确认（动量加速）
4. ADX > `min_adx` 趋势确认

## 参数（`params.yaml`）

| key | 含义 | 默认 |
|---|---|---|
| `fast` | MACD 快线周期 | 12 |
| `slow` | MACD 慢线周期 | 26 |
| `signal` | MACD signal 周期 | 9 |
| `rsi_low` | 做多时 RSI 上限 | 65.0 |
| `rsi_high` | 做空时 RSI 下限 | 35.0 |
| `min_adx` | ADX 下限 | 20.0 |
| `position_pct` | 单次开仓权益占比 | 0.04 |
| `sl_atr_mult` | 止损 ATR 倍数 | 1.8 |
| `tp_atr_mult` | 止盈 ATR 倍数 | 3.0 |
| `slippage_pct` | 单边滑点 | 0.0003 |

## 已知行为

- 实际分批止盈硬编码 `rr_list = [1.0, 2.0]`，`tp_atr_mult` 未直接使用。
