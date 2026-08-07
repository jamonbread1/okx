# RADX — Daily RSI(2) vs ADX(2) Trend Filter

研究策略，regime = `any`，long-only。

## 核心逻辑

**Entry**: `close > SMA(50)` AND `close > EMA(7)` AND `RSI(2) > ADX(2)`
**Exit**: 由 `v3/engine.py` 处理（`RSI(2) < ADX(2)`）—— 详见 `_manage_position`

## 来源

公开社区 BTC/ETH 日线长线策略思路（2025），声称在 2012-2025 日线上
跑赢买入持有但回撤更低。原帖未计入手续费和滑点。

## 参数（`params.yaml`）

| key | 含义 | 默认 |
|---|---|---|
| `sma_period` | SMA 周期 | 50 |
| `ema_period` | EMA 周期 | 7 |
| `rsi_period` | RSI 周期 | 2 |
| `adx_period` | ADX 周期 | 2 |
| `position_pct` | 单次开仓权益占比 | 0.03 |
| `sl_atr_mult` | 止损 ATR 倍数 | 3.0 |
| `tp_r` | TP 倍数（极远，由 RSI<ADX 退出条件控制） | 999.0 |
| `min_confidence` | 最低 confidence | 回退到 `min_open_confidence` |
| `fee_rt` | 单边手续费 | 0.0010 |
| `slippage_pct` | 单边滑点 | 0.0003 |
| `min_atr_pct` | ATR/price 下限 | 0.002 |
| `max_atr_pct` | ATR/price 上限 | 0.12 |
| `long_only` | 仅做多 | true |

## 用法

```bash
python -m v3.run_backtest --bar 1D --only radx --symbols BTC-USDT-SWAP,ETH-USDT-SWAP
```

## 已知行为

- 不在自动策略规划中（仅手动启用）。
- 退出完全依赖 `v3/engine.py` 的 RSI/ADX 条件，`tp_r` 不实际触发。
- 适合 1D 周期验证，短周期噪声太大不建议使用。
