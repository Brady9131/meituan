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

# 在本机启动（稳定版：不依赖 numpy/pandas，避免该环境 segfault）
# 默认端口可直接跑，也可以手动指定，例如 8504
./run_dashboard.sh 8504

# 如果你想更稳（固定缓存目录/禁止 telemetry），也可以用脚本
./run_dashboard.sh

# 该脚本会启动一个不依赖 numpy/pandas 的版本（避免该环境下的 segfault）
```

## 关于“GitHub Pages 为什么不是我们那个交互网站”
你截图的 `github.io/...` 页面通常只是 GitHub Pages 渲染的 `README` 静态内容，**不能直接运行 Streamlit 交互 UI**。真正的网页交互仍需要你在本机运行 `./run_dashboard.sh ...`，然后打开终端里给出的 `http://127.0.0.1:PORT`，或使用下面的云端部署。

## 部署到 Streamlit Community Cloud（推荐，可分享链接）

1. 把本仓库推送到 GitHub（需包含根目录的 `requirements.txt` 与 `app_pyless.py`）。
2. 打开 [Streamlit Community Cloud](https://streamlit.io/cloud)，用 GitHub 账号登录，**New app** → 选择仓库与分支。
3. **Main file path** 填：`app_pyless.py`（不要填 `app.py`）。
4. 部署后你会得到一个 `*.streamlit.app` 公网链接；侧栏上传 CSV 与本地行为一致。
5. （可选）在 Cloud 应用设置里 **Secrets** 添加 DeepSeek 密钥，例如：
   ```toml
   DEEPSEEK_API_KEY = "sk-..."
   ```
   应用会按顺序读取：`DEEPSEEK_API_KEY` 环境变量、侧栏临时粘贴、以及 Cloud 注入的 `st.secrets`。**切勿把密钥写入代码或提交到 Git。**

## 数据格式（推荐）

上传的 CSV 至少需要这些列：
- `user_id`: 用户或候选单元 ID
- `ite`: 个体增量（被处理后相对未处理的 GMV 增量；用于优化）
- `pae`: 价格锚点侵蚀（建议归一化或可比较尺度；用于象限划分）
- `cost`: 本次潜在补贴成本（建议与 `ite` 同一粒度）

> 说明：在当前运行环境中，`Parquet` 读取后端可能导致 Python 进程崩溃，因此仪表盘目前只支持 CSV 上传。

