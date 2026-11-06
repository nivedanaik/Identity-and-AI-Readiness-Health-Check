import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import identity_ai as ia
from data_gen import generate

st.set_page_config(page_title="Identity & AI-Readiness Health Check", layout="wide")


def rec(r):
    v = lambda x: "(missing)" if pd.isna(x) else x
    return f"{v(r.first_name)} {v(r.last_name)} | {v(r.email)} | {v(r.address)}"


@st.cache_data(show_spinner="Generating synthetic data and training the matching model...")
def load(n, seed):
    df = generate(n, seed)
    res = ia.run_matching(df)
    m = ia.source_batch_metrics(df)
    return df, res, m


st.title("Identity & AI-Readiness Health Check")
st.caption("Prototype on 100% synthetic data. Shows how identity quality can be measured, improved with ML, "
           "and turned into plain-language next steps.")

with st.sidebar:
    st.header("Dataset")
    n = st.slider("Synthetic people", 500, 3000, 1500, 250)
    seed = st.number_input("Random seed", 1, 999, 42)
    st.divider()
    st.header("Matching")
    thr = st.slider("ML match threshold", 0.1, 0.95, 0.5, 0.05,
                    help="Higher = fewer false merges, more missed matches.")
    st.divider()
    st.header("Anomaly detection")
    cont = st.slider("Sensitivity (expected share of bad loads)", 0.02, 0.20, 0.08, 0.01)

df, res, metrics = load(n, int(seed))
ev = ia.evaluate(res, thr)
anoms = ia.detect_anomalies(metrics, contamination=cont)
overall, by_src = ia.readiness(df)
f = ia.findings(df, ev, overall, by_src, anoms)

t1, t2, t3, t4, t5 = st.tabs(["Overview", "Identity matching (ML)", "Data quality anomalies (ML)",
                              "AI insights", "About this prototype"])

with t1:
    c = st.columns(4)
    c[0].metric("AI-readiness score", f"{overall.score:.0f}/100", ia.rating(overall.score))
    c[1].metric("Records", f"{len(df):,}")
    c[2].metric("Match recall: rules → ML", f"{ev['ML model']['recall']:.0%}",
                f"{(ev['ML model']['recall'] - ev['Rules only']['recall']) * 100:+.0f} pts vs rules")
    c[3].metric("Flagged source/batch loads", int(anoms.flagged.sum()))
    st.subheader("Readiness by source")
    parts = by_src.reset_index().melt(id_vars="source", value_vars=list(ia.WEIGHTS), var_name="component", value_name="value")
    fig = px.bar(parts, x="source", y="value", color="component", barmode="group",
                 labels={"value": "share (0-1)"}, height=380)
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(by_src.assign(rating=by_src.score.map(ia.rating)).round(2), use_container_width=True)
    st.caption("Score weights: " + ", ".join(f"{k.replace('_', ' ')} {int(v * 100)}%" for k, v in ia.WEIGHTS.items())
               + ". This score is a transparent weighted formula, not a model.")

with t2:
    st.markdown("A gradient-boosted classifier learns which record pairs are the same person. "
                "It is trained on one half of the people and **scored on the other half it has never seen**.")
    rows = []
    for k, v in ev.items():
        rows.append(dict(approach=k, precision=v["precision"], recall=v["recall"], f1=v["f1"],
                         correct_matches=v["tp"], false_merges=v["fp"], missed_matches=v["fn"],
                         resolved_identities=v["identities"]))
    tbl = pd.DataFrame(rows)
    st.dataframe(tbl.style.format({"precision": "{:.1%}", "recall": "{:.1%}", "f1": "{:.1%}"}), use_container_width=True, hide_index=True)
    st.caption(f"Held-out set: {res['test_records']:,} records that truly belong to {res['true_identities']:,} people. "
               "Fewer resolved identities (closer to the true count) means less duplication.")
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure()
        for k in ["Rules only", "ML model"]:
            fig.add_bar(name=k, x=["precision", "recall", "f1"], y=[ev[k]["precision"], ev[k]["recall"], ev[k]["f1"]])
        fig.update_layout(barmode="group", height=340, yaxis_range=[0, 1.02], title="Rules vs ML")
        st.plotly_chart(fig, use_container_width=True)
    with cB:
        imp = res["importances"].head(8).sort_values()
        fig = px.bar(imp, orientation="h", height=340, title="What the model relies on (permutation impact)")
        fig.update_layout(showlegend=False, xaxis_title="share of predictions that flip", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Matches the ML model found that rules missed")
    t = res["test"]
    caught = t[(t.proba >= thr) & (t.rule_match == 0) & (t.label == 1)].sort_values("proba", ascending=False).head(8)
    d = res["d"]
    show = []
    for r in caught.itertuples():
        a, b = d.loc[r.i], d.loc[r.j]
        show.append({"record A": rec(a),
                     "record B": rec(b),
                     "confidence": round(r.proba, 3)})
    st.dataframe(pd.DataFrame(show), use_container_width=True, hide_index=True)
    st.subheader("Low-confidence pairs worth human review")
    gray = t[(t.proba > 0.25) & (t.proba < 0.75)].sort_values("proba").head(8)
    gshow = []
    for r in gray.itertuples():
        a, b = d.loc[r.i], d.loc[r.j]
        gshow.append({"record A": rec(a),
                      "record B": rec(b),
                      "confidence": round(r.proba, 3)})
    st.dataframe(pd.DataFrame(gshow), use_container_width=True, hide_index=True)

with t3:
    st.markdown("An Isolation Forest looks at each source/weekly load across six quality metrics and flags loads that look "
                "unlike the rest, then names the metric that drives the flag.")
    pick = st.selectbox("Metric heatmap", ia.METRIC_COLS, index=0)
    piv = anoms.pivot(index="source", columns="batch", values=pick)
    fig = px.imshow(piv, aspect="auto", color_continuous_scale="Blues", height=300, labels=dict(color=pick))
    flagged = anoms[anoms.flagged]
    for r in flagged.itertuples():
        fig.add_shape(type="rect", x0=list(piv.columns).index(r.batch) - 0.5, x1=list(piv.columns).index(r.batch) + 0.5,
                      y0=list(piv.index).index(r.source) - 0.5, y1=list(piv.index).index(r.source) + 0.5,
                      line=dict(color="crimson", width=3))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red outlines = loads flagged by the model.")
    st.dataframe(flagged.sort_values("anomaly_score", ascending=False)[["source", "batch", "top_driver", "driver_z", "anomaly_score", "records"]]
                 .round(2), use_container_width=True, hide_index=True)

with t4:
    st.markdown("Turns the numbers above into a short brief for a client or stakeholder.")
    key = st.text_input("Optional: Anthropic API key to generate the narrative with Claude", type="password",
                        help="Used only for this request. Without a key, a template summary is shown.")
    if st.button("Generate insights", type="primary"):
        text, used_llm = ia.llm_insights(f, key or None)
        st.info("Written by Claude from the numbers above." if used_llm else "Template summary (no AI narrative).")
        st.markdown(text)
    else:
        st.markdown(ia.template_insights(f))

with t5:
    st.markdown("""
**What is AI here, and what isn't**
- **Identity matching:** supervised ML (gradient-boosted trees) over fuzzy-match features. Compared against a rules-only baseline on held-out people.
- **Anomaly detection:** unsupervised ML (Isolation Forest). The synthetic data contains four planted incidents; the model is not told where they are.
- **Insights:** optional LLM narrative that is told to use only the supplied numbers. Without an API key, a fixed template is used.
- **Readiness score:** a transparent weighted formula, deliberately *not* a model, so it can be explained to a client.

**Limits (read before quoting any number)**
- The data is synthetic and the noise is generated by this app's own rules, so the ML lift shows the approach works, not what Acxiom or any real client would see.
- Real data needs real labels (e.g. reviewed match pairs) to train and to monitor drift.
- No real person, employer, or client data is used anywhere.
""")
