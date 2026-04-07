from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

from dashboard.data import MetricsTable, load_user_metrics, make_sample_data
from dashboard.optimizer_pyless import OptimizationResult, assign_segments, optimize_budget


st.set_page_config(
    page_title="美团补贴提效决策仪表盘",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
      .reportview-container { background: #0b1220; color: #e8eefc; }
      .sidebar .sidebar-content { background: #0f1b33; }
      .stMetric .stMetricLabel { color: #a7b3d6; }
      a { color: #7db2ff; }
      .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px;
      }
      table {
        width: 100%;
        border-collapse: collapse;
      }
      th, td {
        padding: 8px 10px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        text-align: left;
      }
      th { color: #cfe0ff; font-weight: 600; }
      td { color: #e8eefc; }
    </style>
    """,
    unsafe_allow_html=True,
)


SEGMENTS: list[str] = ["Gold", "Addict", "Organic", "Sinking"]
SEG_COLORS = {"Gold": "#4f8cff", "Addict": "#ffb020", "Organic": "#35d07f", "Sinking": "#ff5c7a"}

SEG_DESC = {
    "Gold": "高增益/低侵蚀。优先投放，最大化增量 ROI。",
    "Addict": "高增益/高侵蚀。执行门槛退坡，控制价格心智坍塌风险。",
    "Organic": "低增益/低侵蚀。无券静默，避免对自然增长的“打扰”。",
    "Sinking": "负增益/高侵蚀。断药止损，规避资金净损失。",
}

AGENT_FEEDBACK = {
    "Gold": "多了几块钱，真的够决定一次下单；继续加注。",
    "Addict": "越来越抠但仍有转化；逐步退坡，避免补贴成瘾。",
    "Organic": "没券也会买；不发更干净。",
    "Sinking": "发券反而更亏；拦截并停用。",
}

# (展示名, 默认 Base URL, 常用模型 ID)；「自定义」由用户填写 Base 与模型
LLM_PROVIDER_PRESETS: list[tuple[str, str, list[str]]] = [
    ("DeepSeek", "https://api.deepseek.com", ["deepseek-chat", "deepseek-reasoner"]),
    ("OpenAI", "https://api.openai.com", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]),
    ("通义千问（兼容 OpenAI）", "https://dashscope.aliyuncs.com/compatible-mode/v1", ["qwen-turbo", "qwen-plus", "qwen-max"]),
    ("智谱 GLM", "https://open.bigmodel.cn/api/paas/v4", ["glm-4-flash", "glm-4", "glm-4-plus"]),
    ("Moonshot", "https://api.moonshot.cn/v1", ["moonshot-v1-8k", "moonshot-v1-32k"]),
    ("自定义", "", []),
]


def _openai_compat_chat_completions_url(base_url: str) -> str:
    """不同厂商 OpenAI 兼容接口路径略有差异。"""
    b = base_url.strip().rstrip("/")
    if not b:
        return ""
    if b.endswith("/v1") or "/paas/v4" in b or b.endswith("/v4"):
        return b + "/chat/completions"
    return b + "/v1/chat/completions"


def _num_fmt(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e8:
        return f"{x/1e8:.2f}e8"
    if abs(x) >= 1e4:
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def _mean(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / max(1, len(vals))


def _render_html_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{r.get(h,'')}</td>" for h in headers)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _sample_indices(n: int, max_n: int, seed: int = 7) -> list[int]:
    if n <= max_n:
        return list(range(n))
    rng = random.Random(seed)
    return rng.sample(range(n), max_n)


def _scatter_fig(metrics: MetricsTable, segments: list[str], treated_frac: list[float] | None = None) -> go.Figure:
    n = len(metrics.user_id)
    idx = _sample_indices(n, 20000, seed=7)

    traces: list[go.Scatter] = []
    for seg in SEGMENTS:
        x: list[float] = []
        y: list[float] = []
        custom: list[list[Any]] = []
        for i in idx:
            if segments[i] != seg:
                continue
            x.append(metrics.ite[i])
            y.append(metrics.pae[i])
            frac = treated_frac[i] if treated_frac is not None else 0.0
            custom.append([metrics.user_id[i], metrics.ite[i], metrics.pae[i], metrics.cost[i], frac])

        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=seg,
                marker=dict(size=4, color=SEG_COLORS.get(seg, "#9aa4b2"), opacity=0.78),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "ITE: %{customdata[1]:.3f}<br>"
                    "PAE: %{customdata[2]:.3f}<br>"
                    "Cost: %{customdata[3]:.3f}<br>"
                    "treated_frac: %{customdata[4]:.2f}<extra></extra>"
                ),
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        legend_title_text="Segment",
        margin=dict(l=10, r=10, t=20, b=10),
        height=420,
        xaxis_title="ITE",
        yaxis_title="PAE",
    )
    return fig


st.title("美团补贴效率评估与提效决策仪表盘")
st.caption("基于 `ITE` 与 `PAE` 的象限策略 + 受限预算优化（纯 Python 版，避免 numpy/pandas segfault）。")


st.sidebar.header("数据输入与场景参数")
data_source = st.sidebar.radio("数据来源", ["样例数据（Preview）", "上传指标表（CSV）"])

metrics: MetricsTable | None = None
if data_source == "样例数据（Preview）":
    sample_n = st.sidebar.slider("样例规模", 500, 20000, 2000, step=500)
    # 纯 Python 样例数据
    metrics = make_sample_data(n=sample_n, seed=42)
else:
    uploaded = st.sidebar.file_uploader("上传 CSV（至少包含 user_id, ite, pae, cost）", type=["csv"])
    if uploaded is not None:
        try:
            metrics = load_user_metrics(uploaded)
        except Exception as e:
            st.sidebar.error(f"数据加载失败：{e}")
            metrics = None

st.sidebar.divider()

st.sidebar.subheader("象限分群阈值")
ite_threshold = st.sidebar.number_input("ITE 高低阈值", value=0.0, step=0.05, format="%.2f")
pae_threshold_mode = st.sidebar.selectbox("PAE 阈值模式", ["分位数（建议）", "固定值"])
if pae_threshold_mode == "分位数（建议）":
    pae_q = st.sidebar.slider("PAE 分位数（越高风险越强）", 0.5, 0.95, 0.7, step=0.01)
    pae_mode = "quantile"
    pae_threshold_value = float(pae_q)
else:
    pae_mode = "fixed"
    pae_threshold_value = float(st.sidebar.number_input("PAE 固定阈值", value=0.0, step=0.05, format="%.2f"))

st.sidebar.subheader("受限预算优化")
budget_reduction_pct = st.sidebar.slider("总预算缩减（%）", 0, 30, 10, step=1)
ite_blocking_mode = st.sidebar.radio("负增量处理", ["拦截 ITE < 0（推荐）", "允许 ITE < 0"])
addict_accept_cost_share = st.sidebar.slider(
    "Addict（高增益/高侵蚀）门槛退坡强度：最多允许的成本占比",
    0.05,
    0.9,
    0.35,
    step=0.05,
)


def _secrets_toml_exists() -> bool:
    here = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    home = Path.home() / ".streamlit" / "secrets.toml"
    return here.is_file() or home.is_file()


def _likely_streamlit_community_cloud() -> bool:
    # 云端运行时常见特征（无本地 secrets 文件时仍可向 st.secrets 注入配置）
    return os.environ.get("USER") == "appuser"


def _deepseek_api_key() -> str:
    for env_name in ("DEEPSEEK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"):
        v = str(os.environ.get(env_name, "") or "").strip()
        if v:
            return v
    main = str(st.session_state.get("deepseek_api_key_main") or "").strip()
    if main:
        return main
    side = str(st.session_state.get("deepseek_api_key_sidebar") or "").strip()
    if side:
        return side
    if _secrets_toml_exists() or _likely_streamlit_community_cloud():
        try:
            for k in ("DEEPSEEK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"):
                s = str(st.secrets.get(k, "") or "").strip()
                if s:
                    return s
        except Exception:
            return ""
    return ""


def _deepseek_chat(*, api_key: str, model: str, base_url: str, content: str) -> str:
    """
    OpenAI-compatible chat completion call.
    Return plain text (assistant content).
    """
    url = _openai_compat_chat_completions_url(base_url)
    if not url:
        raise ValueError("请先填写 API Base URL")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严谨的商业分析策略教练。"
                "你要基于给定的 ITE/PAE/成本等信号，输出简洁、可落地的沙盘反馈。"
                "禁止编造与数据不相符的数值；如果信息不足，用相对表述（例如“更高/更低/更稳健”）。",
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.4,
        "max_tokens": 420,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return str(data)


def _roi_sweep_dict(
    metrics: MetricsTable,
    segments: list[str],
    *,
    sweep_max_pct: int,
    addict_accept_cost_share: float,
    ite_blocking_mode_label: str,
) -> dict[str, Any]:
    pcts = list(range(0, int(sweep_max_pct) + 1, 2))
    rois: list[float] = []
    gmvs: list[float] = []
    treated: list[float] = []
    budgets: list[float] = []
    utils: list[float] = []
    mode = "block_negative" if ite_blocking_mode_label.startswith("拦截") else "allow_negative"
    for pct in pcts:
        rr = optimize_budget(
            metrics,
            segments,
            budget_reduction_pct=float(pct),
            addict_accept_cost_share=float(addict_accept_cost_share),
            ite_blocking_mode=mode,
        )
        rois.append(rr.roi)
        gmvs.append(rr.incremental_gmv)
        treated.append(rr.treated_cost)
        budgets.append(rr.budget)
        bu = rr.treated_cost / rr.budget if rr.budget > 1e-12 else 0.0
        utils.append(min(1.0, bu))
    return {
        "pcts": pcts,
        "rois": rois,
        "gmvs": gmvs,
        "treated": treated,
        "budgets": budgets,
        "utils": utils,
        "sweep_max_applied": int(sweep_max_pct),
        "fp": (float(addict_accept_cost_share), ite_blocking_mode_label),
    }


if metrics is None:
    st.info("请先选择数据来源。上传你的 `ite/pae/cost` 指标 CSV 后即可开始交互。")
    st.stop()


segments, pae_thr = assign_segments(
    metrics,
    ite_threshold=float(ite_threshold),
    pae_threshold_mode=pae_mode,  # type: ignore[arg-type]
    pae_threshold_value=float(pae_threshold_value),
)

seg_counts = {s: 0 for s in SEGMENTS}
for s in segments:
    seg_counts[s] += 1

pae_high_rate = sum(1 for s in segments if s in ("Addict", "Sinking")) / max(1, len(segments))

tab_interactive, tab_budget, tab_llm = st.tabs(["交互实验", "受限预算优化结果", "沙盘反馈（LLM）"])

with tab_interactive:
    st.markdown("## ITE/PAE 象限散点（策略可视化）")
    st.plotly_chart(_scatter_fig(metrics, segments), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 样本象限分布")
        for seg in SEGMENTS:
            st.markdown(f"- `{seg}`：{seg_counts[seg]:,}")
    with col2:
        st.markdown("### 阈值口径")
        st.write(f"PAE 阈值（分位数/固定值后）：`{pae_thr:.3f}`")
        st.write(f"ITE 阈值：`{ite_threshold:.3f}`")
    with col3:
        st.markdown("### PAE 风险占比")
        st.metric("PAE 高风险（Addict+Sinking）", f"{pae_high_rate*100:.1f}%")

    st.markdown("## 关键统计概览")
    rows: list[dict[str, Any]] = []
    for seg in SEGMENTS:
        idxs = [i for i, s in enumerate(segments) if s == seg]
        rows.append(
            {
                "Segment": seg,
                "Users": len(idxs),
                "Mean ITE": f"{_mean([metrics.ite[i] for i in idxs]):.4f}",
                "Mean PAE": f"{_mean([metrics.pae[i] for i in idxs]):.4f}",
                "Mean Cost": f"{_mean([metrics.cost[i] for i in idxs]):.4f}",
            }
        )
    st.markdown(
        _render_html_table(
            rows=rows,
            headers=["Segment", "Users", "Mean ITE", "Mean PAE", "Mean Cost"],
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("## 最小替换清单")
    st.markdown(
        """
        1. 用你的 GRF 输出生成 `user_id, ite, pae, cost`（CSV）。
        2. 上传到侧边栏。
        3. 调整预算缩减、PAE 阈值分位数与 Addict 退坡强度，看象限与 ROI 的变化。
        """
    )

with tab_budget:
    st.markdown("## 预算重组与增量 ROI")

    if "opt_result" not in st.session_state:
        st.session_state["opt_result"] = None

    run_opt = st.button("执行预算优化（可能需要几秒）", type="primary")
    if run_opt:
        with st.spinner("执行预算优化（Gold 优先 -> Addict 追加）..."):
            st.session_state["opt_result"] = optimize_budget(
                metrics,
                segments,
                budget_reduction_pct=float(budget_reduction_pct),
                addict_accept_cost_share=float(addict_accept_cost_share),
                ite_blocking_mode="block_negative" if ite_blocking_mode.startswith("拦截") else "allow_negative",
            )
            st.session_state["roi_sweep"] = None  # 参数已变，旧的敏感度曲线作废

    result: OptimizationResult | None = st.session_state.get("opt_result")
    if result is None:
        st.info("当前尚未执行预算优化。请点击上方按钮后再查看 ROI/沙盘反馈。")
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("基线总成本", _num_fmt(result.baseline_cost))
        with m2:
            st.metric("优化后预算", _num_fmt(result.budget))
        with m3:
            st.metric("投放后已使用成本", _num_fmt(result.treated_cost))
        with m4:
            st.metric("增量 ROI", f"{result.roi:.3f}")

        st.markdown(f"### 增量 GMV（来自 ITE 的加权求和）: `{_num_fmt(result.incremental_gmv)}`")

        _sweep_max = st.selectbox("扫描最大预算缩减（%）", [30, 50, 70, 90], index=1, key="roi_sweep_max_pct")
        _max_i = int(_sweep_max)
        _sweep_fp = (float(addict_accept_cost_share), ite_blocking_mode)
        sweep = st.session_state.get("roi_sweep")
        sweep_stale = (
            sweep is None
            or "budgets" not in sweep
            or sweep.get("sweep_max_applied") != _max_i
            or sweep.get("fp") != _sweep_fp
        )
        if sweep_stale:
            with st.spinner("正在生成敏感度曲线…"):
                st.session_state["roi_sweep"] = _roi_sweep_dict(
                    metrics,
                    segments,
                    sweep_max_pct=_max_i,
                    addict_accept_cost_share=float(addict_accept_cost_share),
                    ite_blocking_mode_label=ite_blocking_mode,
                )
            sweep = st.session_state["roi_sweep"]

        if sweep:
            fig_curve = make_subplots(specs=[[{"secondary_y": True}]])
            fig_curve.add_trace(
                go.Scatter(
                    x=sweep["pcts"],
                    y=sweep["rois"],
                    name="增量 ROI",
                    mode="lines+markers",
                    line=dict(color="#7db2ff", width=3),
                    marker=dict(size=6),
                ),
                secondary_y=False,
            )
            fig_curve.add_trace(
                go.Scatter(
                    x=sweep["pcts"],
                    y=sweep["gmvs"],
                    name="增量 GMV",
                    mode="lines+markers",
                    line=dict(color="#35d07f", width=2),
                    marker=dict(size=5),
                ),
                secondary_y=True,
            )
            fig_curve.update_layout(
                template="plotly_dark",
                height=440,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=28, b=10),
            )
            fig_curve.update_xaxes(title_text="总预算缩减（%）", dtick=5)
            fig_curve.update_yaxes(title_text="增量 ROI", secondary_y=False, gridcolor="rgba(255,255,255,0.08)")
            fig_curve.update_yaxes(title_text="增量 GMV（ITE×投放强度）", secondary_y=True, gridcolor="rgba(255,255,255,0.08)")
            st.plotly_chart(fig_curve, use_container_width=True)

            fig_budget_cmp = go.Figure()
            fig_budget_cmp.add_trace(
                go.Scatter(
                    x=sweep["pcts"],
                    y=sweep["budgets"],
                    name="可用预算上限",
                    mode="lines+markers",
                    line=dict(color="#9aa4b2", width=2, dash="dash"),
                    marker=dict(size=5),
                )
            )
            fig_budget_cmp.add_trace(
                go.Scatter(
                    x=sweep["pcts"],
                    y=sweep["treated"],
                    name="实际投放成本",
                    mode="lines+markers",
                    fill="tonexty",
                    fillcolor="rgba(255,176,32,0.15)",
                    line=dict(color="#ffb020", width=3),
                    marker=dict(size=6),
                )
            )
            fig_budget_cmp.update_layout(
                template="plotly_dark",
                height=360,
                xaxis_title="总预算缩减（%）",
                yaxis_title="金额",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
                margin=dict(l=10, r=10, t=28, b=10),
            )
            st.plotly_chart(fig_budget_cmp, use_container_width=True)

            fig_util = go.Figure(
                data=[
                    go.Scatter(
                        x=sweep["pcts"],
                        y=sweep["utils"],
                        name="预算利用率（实际消耗/可用预算）",
                        mode="lines+markers",
                        fill="tozeroy",
                        line=dict(color="#7db2ff", width=2),
                        marker=dict(size=5),
                    )
                ]
            )
            fig_util.update_layout(
                template="plotly_dark",
                height=300,
                xaxis_title="总预算缩减（%）",
                yaxis_title="利用率",
                yaxis_tickformat=".0%",
                yaxis_range=[
                    0,
                    max(0.1, min(1.05, (max(sweep["utils"]) if sweep["utils"] else 0) * 1.15 + 0.03)),
                ],
                margin=dict(l=10, r=10, t=28, b=10),
            )
            st.plotly_chart(fig_util, use_container_width=True)

        st.markdown("#### 当前结果：各象限成本与增量占比")
        cbar1, cbar2 = st.columns(2)
        cost_segments = sorted(
            result.segment_cost_share.keys(),
            key=lambda k: result.segment_cost_share[k],
            reverse=True,
        )
        cost_vals = [result.segment_cost_share[s] for s in cost_segments]
        with cbar1:
            fig_cost = go.Figure(
                data=[
                    go.Bar(
                        x=cost_segments,
                        y=cost_vals,
                        marker_color=[SEG_COLORS.get(s, "#9aa4b2") for s in cost_segments],
                        showlegend=False,
                        text=[f"{v*100:.1f}%" for v in cost_vals],
                        textposition="outside",
                    )
                ]
            )
            fig_cost.update_layout(
                template="plotly_dark",
                title="投放成本占比（按象限）",
                yaxis_tickformat=".0%",
                margin=dict(l=10, r=10, t=50, b=10),
                height=340,
            )
            st.plotly_chart(fig_cost, use_container_width=True)

        inc_segments = sorted(
            result.segment_increment_share.keys(),
            key=lambda k: result.segment_increment_share[k],
            reverse=True,
        )
        inc_vals = [result.segment_increment_share[s] for s in inc_segments]
        with cbar2:
            fig_inc = go.Figure(
                data=[
                    go.Bar(
                        x=inc_segments,
                        y=inc_vals,
                        marker_color=[SEG_COLORS.get(s, "#9aa4b2") for s in inc_segments],
                        showlegend=False,
                        text=[f"{v*100:.1f}%" for v in inc_vals],
                        textposition="outside",
                    )
                ]
            )
            fig_inc.update_layout(
                template="plotly_dark",
                title="增量 GMV 占比（按象限）",
                yaxis_tickformat=".0%",
                margin=dict(l=10, r=10, t=50, b=10),
                height=340,
            )
            st.plotly_chart(fig_inc, use_container_width=True)

with tab_llm:
    st.markdown("## 沙盘反馈（LLM）")
    st.caption("与「交互实验」「受限预算优化结果」并列；需先在 **受限预算优化结果** 中执行一次预算优化。")

    use_deepseek = st.toggle("启用 LLM 生成反馈", value=False)

    st.text_input("API Key", placeholder="与所选服务商一致", key="deepseek_api_key_sidebar")

    _provider_labels = [p[0] for p in LLM_PROVIDER_PRESETS]
    _provider_map = {p[0]: (p[1], p[2]) for p in LLM_PROVIDER_PRESETS}
    llm_provider = st.selectbox("API 服务商", _provider_labels, key="llm_provider_choice")
    _base_def, _model_opts = _provider_map[llm_provider]

    if llm_provider == "自定义":
        deepseek_base_url = st.text_input(
            "API Base URL",
            placeholder="https://api.example.com 或含 /v1 的兼容地址",
            key="llm_base_custom",
        )
        deepseek_model = st.text_input("模型 ID", placeholder="与服务商文档一致", key="llm_model_custom_only")
    else:
        deepseek_base_url = st.text_input(
            "API Base URL",
            value=_base_def,
            key=f"llm_base_{llm_provider}",
        )
        _model_choices = _model_opts + ["其他（手动输入）"]
        _model_pick = st.selectbox("模型", _model_choices, key=f"llm_model_pick_{llm_provider}")
        if _model_pick == "其他（手动输入）":
            deepseek_model = st.text_input("自定义模型 ID", key=f"llm_model_typed_{llm_provider}")
        else:
            deepseek_model = _model_pick

    st.subheader("沙盘金额口径")
    deepseek_amount_mode = st.radio(
        "补贴/优惠券金额用于沙盘的来源",
        ["使用优化估算", "手动指定固定金额"],
        index=0,
        horizontal=True,
    )
    if deepseek_amount_mode == "手动指定固定金额":
        deepseek_fixed_amount = st.number_input(
            "手动固定补贴金额（RMB）",
            value=5.0,
            step=0.5,
            format="%.1f",
            key="deepseek_fixed_amount_llm",
        )
    else:
        st.caption("选择「使用优化估算」时，沙盘金额以预算优化结果为准，无需填写固定金额。")
        deepseek_fixed_amount = float(st.session_state.get("deepseek_fixed_amount_llm", 5.0))

    st.markdown("---")
    st.markdown("### 两段式沙盘输出（按象限）")

    result_llm: OptimizationResult | None = st.session_state.get("opt_result")
    if result_llm is None:
        st.info("请先到 **「受限预算优化结果」** 标签页，点击 **「执行预算优化」** 后再回到本页查看沙盘。")
    else:
        treated_idxs = [i for i, f in enumerate(result_llm.treated_frac) if f > 1e-9]
        if not treated_idxs:
            st.warning("当前参数下未选择任何可投放单元。请在「受限预算优化结果」中放宽预算或调整阈值。")
        else:
            for seg in SEGMENTS:
                seg_idxs = [i for i in treated_idxs if segments[i] == seg]
                if not seg_idxs:
                    continue
                top3 = sorted(seg_idxs, key=lambda i: metrics.ite[i], reverse=True)[:3]
                st.markdown(f"#### {seg}（处理单元：{len(seg_idxs):,}）")
                ids = [int(metrics.user_id[i]) for i in top3]

                fallback = AGENT_FEEDBACK[seg]

                if use_deepseek:
                    api_key = _deepseek_api_key()
                    if not api_key:
                        st.info(
                            "未检测到 API Key（上方填写或环境变量 `DEEPSEEK_API_KEY` / `LLM_API_KEY` / `OPENAI_API_KEY`）。已使用规则化反馈。"
                        )
                        st.markdown(fallback)
                        st.caption(f"示例候选：{ids}")
                        continue

                    persona_map = {
                        "Gold": "你是 Gold 高潜摇摆客，追求生活品质；这次拿到更合适的优惠后会立刻下单。",
                        "Addict": "你是 Addict 中度依赖老客，习惯用美团；即使优惠略有变化也会接受并完成下单。",
                        "Organic": "你是 Organic 绝对刚需客；就算现金补贴为 0 也最终会为饱腹完成下单。",
                        "Sinking": "你是 Sinking 重度羊毛党；不合理的发券只会让你更挑剔并可能流失。",
                    }
                    case_name_map = {
                        "Gold": "表型 2: Gold 高潜摇摆客",
                        "Addict": "表型 4: Addict 中度依赖老客",
                        "Organic": "表型 3: Organic 绝对刚需客",
                        "Sinking": "表型 1: Sinking 重度羊毛党",
                    }

                    treated_summary_blocks: list[str] = []
                    for i in top3:
                        avg_reduce_amount = float(metrics.cost[i])
                        f = float(result_llm.treated_frac[i])
                        optimal_subsidy = avg_reduce_amount * f
                        send_amount = optimal_subsidy
                        if deepseek_amount_mode == "手动指定固定金额":
                            send_amount = float(deepseek_fixed_amount)
                        treated_summary_blocks.append(
                            "\n".join(
                                [
                                    f"- user_id: {int(metrics.user_id[i])}",
                                    f"  ITE: {metrics.ite[i]:.4f}",
                                    f"  PAE: {metrics.pae[i]:.4f}",
                                    f"  avg_reduce_amount(历史均值): {avg_reduce_amount:.4f}",
                                    f"  optimal_subsidy(估算): {optimal_subsidy:.4f}",
                                    f"  send_amount(沙盘口径): {send_amount:.4f}",
                                ]
                            )
                        )

                    prompt = "\n".join(
                        [
                            "你将扮演 stage9 的两段式沙盘：先 Strategy Agent 决定补贴/文案，再 User Agent 作为用户给出内心 OS 和决策。",
                            "要求：",
                            "1) 输出必须严格分为两部分：`Strategy Agent` 与 `User Agent`。",
                            "2) `Strategy Agent` 必须输出：`补贴金额`（保留 1 位小数）与 `文案`（40字以内，不要出现“ITE/PAE”）。",
                            "3) `User Agent` 必须输出：`真实内心OS`（30字以内）与 `决定`（仅输出【转化成功】或【转化失败】）。",
                            "4) 不要编造具体与输入无关的数值；若信息不足，用“更高/更低/更稳健”等相对表述。",
                            "",
                            f"当前象限/策略:{seg}",
                            f"策略目标:{SEG_DESC[seg]}",
                            f"对应表型:{case_name_map.get(seg, seg)}",
                            f"用户人设:{persona_map.get(seg, '')}",
                            "",
                            "候选单元（Top3 by ITE）：",
                            *treated_summary_blocks,
                            "",
                            "规则化模板反馈（用于风格对齐，允许改写）：",
                            fallback,
                            "",
                            "请在 Strategy Agent 中使用候选单元对应的 send_amount 作为补贴/优惠券金额。",
                            "",
                            "附加策略约束（用于 Strategy Agent）：",
                            "A) 若 optimal_subsidy==0：避免提“现金红包”，优先使用免配送费/权益/服务质量等非现金口径；但 Organic 人设可仍然保持“会下单”。",
                            "B) 若 seg 为 Gold：强调“更合适的大额福利/立刻下单”。",
                            "C) 若 seg 为 Addict：强调“门槛退坡/更稳健的优惠节奏”，控制价格心智侵蚀（PAE）。",
                            "D) PAE 越高，文案越偏向“节制、稳定、弱化红包依赖”。",
                            "",
                            "补充约束（对齐你的输入）：",
                            "本次发券/补贴金额口径以 send_amount 为准（若你选择了手动固定金额，则 send_amount 为该固定值）。",
                        ]
                    )

                    with st.spinner("调用 LLM 生成两段式沙盘反馈…"):
                        try:
                            text = _deepseek_chat(
                                api_key=api_key,
                                model=str(deepseek_model).strip(),
                                base_url=str(deepseek_base_url).strip(),
                                content=prompt,
                            )
                            st.markdown(text)
                            st.caption(f"示例候选：{ids}")
                        except Exception as e:
                            st.warning(f"LLM 调用失败：{e}（已回退规则化反馈）")
                            st.markdown(fallback)
                            st.caption(f"示例候选：{ids}")
                else:
                    st.markdown(fallback)
                    st.caption(f"示例候选：{ids}")
