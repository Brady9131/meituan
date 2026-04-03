from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data import load_user_metrics, make_sample_data
from dashboard.optimizer import assign_segments, optimize_budget


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
      .stProgress > div > div { background-color: #4f8cff; }
      a { color: #7db2ff; }
      .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def _num_fmt(x: float) -> str:
    if abs(x) >= 1e8:
        return f"{x/1e8:.2f}e8"
    if abs(x) >= 1e4:
        return f"{x:,.0f}"
    return f"{x:,.2f}"


@st.cache_data(show_spinner=False)
def load_sample(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    return make_sample_data(n=n, seed=seed)


with st.container():
    st.title("美团补贴效率评估与提效决策仪表盘")
    st.caption("基于 `ITE` 与 `PAE` 的象限策略 + 受限预算优化（分数背包近似）。")


st.sidebar.header("数据输入与场景参数")

data_source = st.sidebar.radio("数据来源", ["样例数据（Preview）", "上传指标表（推荐）"])

df: pd.DataFrame | None = None
if data_source == "样例数据（Preview）":
    sample_n = st.sidebar.slider("样例规模", 500, 20000, 2000, step=500)
    df = load_sample(n=sample_n, seed=42)
else:
    uploaded = st.sidebar.file_uploader("上传 CSV/Parquet（至少包含 user_id, ite, pae, cost）", type=["csv", "parquet"])
    if uploaded is not None:
        try:
            with st.spinner("加载指标表..."):
                df = load_user_metrics(uploaded)
        except Exception as e:
            st.sidebar.error(f"数据加载失败：{e}")
            df = None


st.sidebar.divider()
st.sidebar.subheader("象限分群阈值")

ite_threshold = st.sidebar.number_input("ITE 高低阈值", value=0.0, step=0.05, format="%.2f")
pae_threshold_mode = st.sidebar.selectbox("PAE 阈值模式", ["分位数（建议）", "固定值"])
if pae_threshold_mode == "分位数（建议）":
    pae_q = st.sidebar.slider("PAE 分位数（越高风险越强）", 0.5, 0.95, 0.7, step=0.01)
    pae_threshold_value: float = float(pae_q)
    pae_mode = "quantile"
else:
    pae_val = st.sidebar.number_input("PAE 固定阈值", value=0.0, step=0.05, format="%.2f")
    pae_threshold_value = float(pae_val)
    pae_mode = "fixed"


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

if df is None:
    st.info("请先选择数据来源。上传你的 `ite/pae` 指标表后即可开始交互。")
    st.stop()


with st.spinner("象限分群与指标计算中..."):
    df_seg = assign_segments(
        df,
        ite_threshold=float(ite_threshold),
        pae_threshold_mode=pae_mode,  # type: ignore[arg-type]
        pae_threshold_value=float(pae_threshold_value),
    )


col_a, col_b, col_c = st.columns(3)
with col_a:
    seg_counts = df_seg["segment"].value_counts().to_dict()
    st.markdown(f"### 样本象限分布")
    for seg in ["Gold", "Addict", "Organic", "Sinking"]:
        st.markdown(f"- `{seg}`：{seg_counts.get(seg, 0):,}")
with col_b:
    st.markdown("### 象限口径说明（与你报告对齐）")
    st.markdown(
        "\n".join(
            [
                f"- **Gold**：高 ITE，低 PAE（优先投放）",
                f"- **Addict**：高 ITE，高 PAE（门槛退坡）",
                f"- **Organic**：低 ITE，低 PAE（无券静默）",
                f"- **Sinking**：低 ITE，高 PAE（断药止损）",
            ]
        )
    )
with col_c:
    pae_high_rate = float((df_seg["segment"].isin(["Addict", "Sinking"])).mean())
    st.markdown("### PAE 风险占比")
    st.metric("PAE 高风险（Addict+Sinking）", f"{pae_high_rate*100:.1f}%")


left, right = st.columns([2.1, 1.4])

with left:
    st.markdown("## ITE/PAE 象限散点（策略可视化）")
    fig = px.scatter(
        df_seg.sample(min(len(df_seg), 20000), random_state=7),
        x="ite",
        y="pae",
        color="segment",
        opacity=0.75,
        size_max=4,
        hover_data=["user_id", "ite", "pae", "cost"],
        color_discrete_map={"Gold": "#4f8cff", "Addict": "#ffb020", "Organic": "#35d07f", "Sinking": "#ff5c7a"},
    )
    fig.update_layout(
        template="plotly_dark",
        legend_title_text="Segment",
        margin=dict(l=10, r=10, t=20, b=10),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("## 关键统计概览")
    summary = (
        df_seg.groupby("segment")
        .agg(
            users=("user_id", "count"),
            mean_ite=("ite", "mean"),
            mean_pae=("pae", "mean"),
            mean_cost=("cost", "mean"),
        )
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("---")

tab1, tab2 = st.tabs(["决策象限与解释", "受限预算优化结果"])

with tab1:
    st.markdown("## 策略矩阵（Strategy Matrix）")
    matrix_rows = []
    for seg in ["Gold", "Addict", "Organic", "Sinking"]:
        matrix_rows.append(
            {
                "Segment": seg,
                "核心目标": SEG_DESC[seg].split("。")[0] + "。",
                "投放动作": {
                    "Gold": "Uplift Maximization（优先加注）",
                    "Addict": "Threshold Receding（门槛退坡）",
                    "Organic": "Zero-Subsidy（无券静默）",
                    "Sinking": "Blocking（断药止损）",
                }[seg],
                "业务洞察": SEG_DESC[seg],
            }
        )
    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

    st.markdown("## 学术严谨性：关键指标对齐")
    st.markdown(
        r"""
        - **因果估计的核心**：个体增量效应 `ITE` 通常可写作
          `ITE_i = Y_i(1) - Y_i(0)`（接受补贴 vs 未接受补贴的潜在结果差）。
        - **价格心智侵蚀（PAE）**：用于度量补贴干预对用户长期“价格锚点”的负向影响风险。
        - **策略划分**：将 `ITE`（增量贡献）与 `PAE`（侵蚀风险）映射到四象限，从而实现“增量 ROI”和“心智保护”的双目标权衡。
        - **预算约束**：通过分数背包近似，将全局预算重组到更高边际收益的候选单元上。
        """
    )

with tab2:
    with st.spinner("执行预算优化（Gold 优先 -> Addict 追加）..."):
        result = optimize_budget(
            df_seg,
            budget_reduction_pct=float(budget_reduction_pct),
            addict_accept_cost_share=float(addict_accept_cost_share),
            ite_blocking_mode="block_negative" if ite_blocking_mode.startswith("拦截") else "allow_negative",
        )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("基线总成本", f"{_num_fmt(result.baseline_cost)}")
    with m2:
        st.metric("优化后预算", f"{_num_fmt(result.budget)}")
    with m3:
        st.metric("投放后已使用成本", f"{_num_fmt(result.treated_cost)}")
    with m4:
        st.metric("增量 ROI", f"{result.roi:.3f}")

    st.markdown(f"### 增量 GMV（来自 ITE 的加权求和）: `{_num_fmt(result.incremental_gmv)}`")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("## 分段成本占比")
        seg_cost_df = pd.DataFrame(
            {"segment": list(result.segment_cost_share.keys()), "cost_share": list(result.segment_cost_share.values())}
        ).sort_values("cost_share", ascending=False)
        fig_cost = px.bar(
            seg_cost_df,
            x="segment",
            y="cost_share",
            color="segment",
            color_discrete_map={"Gold": "#4f8cff", "Addict": "#ffb020", "Organic": "#35d07f", "Sinking": "#ff5c7a"},
            template="plotly_dark",
        )
        fig_cost.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10), height=320)
        st.plotly_chart(fig_cost, use_container_width=True)

    with g2:
        st.markdown("## 分段增量占比")
        seg_inc_df = pd.DataFrame(
            {
                "segment": list(result.segment_increment_share.keys()),
                "inc_share": list(result.segment_increment_share.values()),
            }
        ).sort_values("inc_share", ascending=False)
        fig_inc = px.bar(
            seg_inc_df,
            x="segment",
            y="inc_share",
            color="segment",
            color_discrete_map={"Gold": "#4f8cff", "Addict": "#ffb020", "Organic": "#35d07f", "Sinking": "#ff5c7a"},
            template="plotly_dark",
        )
        fig_inc.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10), height=320)
        st.plotly_chart(fig_inc, use_container_width=True)

    st.markdown("---")
    st.markdown("## 交互式“沙盘反馈”（规则化占位，可接入你的 Multi-Agent）")
    # Sample a few treated candidates per segment for a qualitative summary
    treated = result.treated_df[result.treated_df["treated_frac"] > 1e-9].copy()
    if treated.empty:
        st.warning("当前参数下未选择任何可投放单元。请降低预算约束或放宽阈值。")
    else:
        for seg in ["Gold", "Addict", "Organic", "Sinking"]:
            seg_df = treated[treated["segment"] == seg]
            if seg_df.empty:
                continue
            top = seg_df.sort_values("ite", ascending=False).head(3)
            st.markdown(f"### {seg}（处理单元：{len(seg_df):,}）")
            st.markdown(AGENT_FEEDBACK[seg])
            ids = top["user_id"].astype(int).tolist()
            st.caption(f"示例候选：{ids}")

    st.markdown(
        """
        说明：这里先用“规则化模板”生成反馈，保证交互闭环。你后续如果把 GRF/PAE 的解释结果（例如 SHAP 交互项、Qini 曲线、负增量占比）导出成文件，
        我可以把本模块升级为“真实实验日志驱动”的 Multi-Agent 仿真。
        """
    )


st.markdown("---")

# 把论文中的关键图表收拢成一个“报告视图”Tab，方便现场展示
tab_report = st.tabs(["报告视图（与论文对齐）"])[0]
assets_base = "/Users/bradyzhao/.cursor/projects/Users-bradyzhao-Desktop-Meituan-Analysis/assets"

with tab_report:
    st.markdown("### 1. 偏差校准与 ATT 结果")
    st.image(
        f"{assets_base}/image-d1c4e177-c2d6-4b25-a0e3-c8faa5f9cca5.png",
        caption="Average Treatment Effect on the Treated (ATT)：在 PSM 匹配后，补贴显著提升了平均 GMV。",
        use_column_width=True,
    )

    st.markdown("### 2. 历史补贴分布与安全边界")
    st.image(
        f"{assets_base}/image-7e530811-3e0b-4282-a875-9039e5c9532c.png",
        caption="历史人均补贴分布呈长尾，P95≈7.63 RMB 作为业务安全上限。",
        use_column_width=True,
    )

    st.markdown("### 3. PAE 机制拆解（全局与异质性）")
    c1, c2 = st.columns(2)
    with c1:
        st.image(
            f"{assets_base}/image-ad5f2e20-690c-475d-bb6b-000e5d6d81d7.png",
            caption="Top 驱动因子：A_pay_level 与 transacted_bu_count 等是 PAE 的核心决定变量。",
            use_column_width=True,
        )
    with c2:
        st.image(
            f"{assets_base}/image-41f2d27b-9266-427f-80ba-315852133f19.png",
            caption="SHAP 全局影响：高 A_pay_level 人群在 PAE 上呈明显正向偏移。",
            use_column_width=True,
        )

    st.markdown("### 4. A_pay_level × transacted_bu_count 的调节效应")
    st.image(
        f"{assets_base}/image-d1f745ae-2845-4f5f-bdb9-5098f2c10ab0.png",
        caption="跨业务覆盖越高时，高消费等级人群的价格心智侵蚀被显著稀释。",
        use_column_width=True,
    )

    st.markdown("### 5. ITE 分布与负增量识别")
    st.image(
        f"{assets_base}/image-0dea67ed-42d4-42bc-b1e5-c768f2bfa058.png",
        caption="个体增量效应（ITE）分布：左侧暴露出需强制拦截的负 ROI 区间。",
        use_column_width=True,
    )

    st.markdown("### 6. ITE × PAE 决策矩阵（Gold / Addict / Organic / Sinking）")
    st.image(
        f"{assets_base}/image-f9b8a9ce-9bf6-41b0-87fb-b7a1bf354111.png",
        caption="Final Subsidy Decision Matrix：与你在交互散点中实时看到的象限逻辑完全一致。",
        use_column_width=True,
    )

    st.markdown("### 7. 历史补贴长尾与极值截断（P95 / P99）")
    st.image(
        f"{assets_base}/image-67b2238e-9a57-4c3f-a0a2-f579fb42450f.png",
        caption="进一步展示补贴极值的长尾特征，为 P99 截断与黑产防御提供依据。",
        use_column_width=True,
    )

    st.markdown("### 8. 分段预算重组（受限背包）")
    st.image(
        f"{assets_base}/image-1a766591-e768-4b76-bf92-0fab1bb2c367.png",
        caption="Counterfactual Budget Reallocation：Gold 获得 +20% 资本注入，Sinking 被完全切断。",
        use_column_width=True,
    )

    st.markdown("### 9. Qini 曲线：全局极值与模型增益")
    st.image(
        f"{assets_base}/image-aac6bfb5-2106-4707-b9bc-9b1a962abfef.png",
        caption="Qini 曲线显示在前约 28% 高 ITE 用户时达到 GMV 增量峰值。",
        use_column_width=True,
    )

    st.markdown("### 10. 表型聚类与多智能体沙盘")
    st.image(
        f"{assets_base}/image-e3d579c1-0c21-4e77-842b-c71d14fea3b5.png",
        caption="PCA + 聚类得到的 4/5 类行为表型，可与 LLM Multi-Agent 仿真一一对照。",
        use_column_width=True,
    )

st.markdown("---")
st.markdown("## 你接下来需要做的最小替换")
st.markdown(
    """
    1. 用你的因果森林（GRF）产出生成一个表：`user_id, ite, pae, cost`（或同粒度的候选单元）。
    2. 通过侧边栏上传该文件。
    3. 调整预算缩减、PAE 阈值分位数和 Addict 退坡强度，观察策略象限与 ROI 的变化。
    """
)

