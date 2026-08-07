# MING — Donchian + MACD 1D 双触发共振

1D 周期专用策略。

## 设计目标

在日线频率上整合「价格结构突破」和「动量方向一致」两个独立信号源，**必须同时触发**才开仓，防止单信号假突破。

| 组件 | 职责 |
|---|---|
| `don` | 确认「价格突破前 N 日区间」+ 量能 + 影线过滤 + ADX 趋势 |
| `macd` | 确认「动量方向 + hist 扩张」+ ADX 趋势 |
| 加权评分 | `weighted = don_score * w_don + macd_score * w_macd` |
| 硬双触发 | `don_long AND macd_long` / `don_short AND macd_short` |

完整设计说明见项目根 README §5.2。

## 用法

```bash
# 1D 单独跑 ming
python -m v3.run_backtest --bar 1D --only ming

# 1D 全策略（auto plan 默认只跑 ming）
python -m v3.run_backtest --bar 1D
```

## 参数（`params.yaml`）

详细参数分组见 `v3/strategies/ming/params.yaml`（裸 key，无 `ming_` 前缀）。
v3-v5 调参 changelog 详见项目根 README §5.2.8-5.2.10。
