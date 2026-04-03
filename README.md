# Meituan Subsidy Decision Dashboard

用 Python + Streamlit 做的交互式决策仪表盘骨架，支持：
- 上传“每用户/每候选单元”的 `user_id, ite, pae, cost` 指标表
- 根据 `ITE` 与 `PAE` 的象限划分 `Gold / Addict / Organic / Sinking`
- 进行受限预算优化（Gold 优先，其次 Addict；负增量默认拦截）
- 展示 ITE/PAE 分布、象限散点、预算分配与 ROI

## 运行

```bash
# 推荐：先激活虚拟环境，然后用 streamlit 启动（不要用 `python app.py`）
source .venv/bin/activate

# 在本机启动
streamlit run app.py

# 如果你想更稳（固定缓存目录/禁止 telemetry），也可以用脚本
./run_dashboard.sh

# 该脚本会启动一个不依赖 numpy/pandas 的版本（避免该环境下的 segfault）
```

## 数据格式（推荐）

上传的 CSV 至少需要这些列：
- `user_id`: 用户或候选单元 ID
- `ite`: 个体增量（被处理后相对未处理的 GMV 增量；用于优化）
- `pae`: 价格锚点侵蚀（建议归一化或可比较尺度；用于象限划分）
- `cost`: 本次潜在补贴成本（建议与 `ite` 同一粒度）

> 说明：在当前运行环境中，`Parquet` 读取后端可能导致 Python 进程崩溃，因此仪表盘目前只支持 CSV 上传。

