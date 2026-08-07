# DON — Donchian Breakout（唐奇安通道突破）

趋势跟踪策略，regime = `trend`。

## 核心逻辑

1. 价格突破前 `period` 根 K 线的最高价 → 做多
2. 价格跌破前 `period` 根 K 线的最低价 → 做空
3. 放量确认（`vol_ratio`）
4. 影线过滤（避免假突破）
5. ADX 过滤（`min_adx`）

## 参数（`params.yaml`）

| key | 含义 | 默认 |
|---|---|---|
| `period` | 唐奇安通道周期 | 28 |
| `min_adx` | ADX 下限，趋势太弱不开仓 | 25.0 |
| `vol_ratio` | 放量倍数下限 | 1.3 |
| `max_wick_ratio` | 影线/实体比上限 | 2.0 |
| `position_pct` | 单次开仓权益占比 | 0.04 |
| `sl_atr_mult` | 止损 = max(ATR × 此值, price × 0.5%) | 1.8 |
| `tp_atr_mult` | 止盈 ATR 倍数（仅作参考） | 3.5 |
| `slippage_pct` | 单边滑点 | 0.0004 |
| `fee_rt` | 单边手续费率 | 0.0007 |
| `short_penalty` | 做空仓位乘数 | 0.7 |
| `calc_window` | EWM 指标计算窗口 | 250 |

## 跨策略共享字段

以下字段在 `config.yaml → strategy:` 段下管理，不在此文件里：

- `min_open_confidence`（开仓最低 confidence）
- `position_timeout_sec`（持仓超时）
- `signal_bar`、`main_bar`、`_main_bar`（信号周期）
- `regime_*`（regime 阈值）
- `htf_*`（高周期过滤）
- `funding_*`（资金费）

## 用法

```bash
# 单策略回测
python -m v3.run_backtest --bar 1H --only don

# 组合回测（config.yaml 启用 don）
python -m v3.run_backtest --bar 1H

# 命令行临时改一个参数
python -m v3.run_backtest --bar 1H --only don --params don:period=20

# 整套覆盖（从另一个 yaml）
python -m v3.run_backtest --bar 1H --only don --extra-config my_don.yaml
```

## 已知行为

- `tp_atr_mult` 在 `__init__` 读取但 `generate()` 未直接使用（止盈由
  硬编码的 `rr_list=[1.0, 2.0, 3.0]` 决定）。如果想让 `tp_atr_mult` 真正
  影响止盈，需要在 `generate()` 里改写 `rr_list` 计算逻辑。
- 做空需要 `vol_ratio >= self.vol_ratio + 0.3`，比做多要求更高（量价配合）。
