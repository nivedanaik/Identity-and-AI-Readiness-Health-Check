"""ML identity matching, anomaly detection, AI-readiness scoring, and insight generation."""
import os
import re
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler

from data_gen import NICKNAMES, REF_DATE

_NICK_TO_CANON = {}
for canon, nicks in NICKNAMES.items():
    _NICK_TO_CANON[canon.lower()] = canon.lower()
    for n in nicks:
        _NICK_TO_CANON[n.lower()] = canon.lower()

_ABBR = {"street": "st", "avenue": "ave", "road": "rd", "drive": "dr", "lane": "ln", "boulevard": "blvd", "court": "ct"}

FEATURES = ["jw_first", "jw_last", "nick_eq", "first_initial_eq", "addr_sim", "zip_eq", "city_eq",
            "email_exact", "email_local_sim", "phone_exact", "email_missing", "phone_missing", "same_source"]


# ---------- normalization ----------
def _digits(p):
    if pd.isna(p):
        return ""
    d = re.sub(r"\D", "", str(p))
    return d[-10:] if len(d) >= 10 else ""


def _addr(a):
    if pd.isna(a) or not a:
        return ""
    a = re.sub(r"[^\w\s]", " ", a.lower())
    return " ".join(_ABBR.get(t, t) for t in a.split())


def _prep(df):
    d = df.reset_index(drop=True).copy()
    d["f"] = d.first_name.fillna("").str.lower()
    d["l"] = d.last_name.fillna("").str.lower()
    d["fc"] = d.f.map(lambda x: _NICK_TO_CANON.get(x, x))
    d["em"] = d.email.fillna("").str.strip().str.lower()
    d["em_local"] = d.em.str.split("@").str[0]
    d["ph"] = d.phone.map(_digits)
    d["ad"] = d.address.map(_addr)
    return d


# ---------- candidate generation (blocking) ----------
def candidate_pairs(d, max_block=40):
    blocks = defaultdict(list)
    for i, r in enumerate(d.itertuples()):
        if r.l:
            blocks[("ln", r.l[:3], r.f[:1])].append(i)
        if r.zip:
            blocks[("zip", r.zip)].append(i)
        if r.em_local:
            blocks[("em", r.em_local[:6])].append(i)
        if r.ph:
            blocks[("ph", r.ph)].append(i)
    pairs = set()
    for ids in blocks.values():
        if 1 < len(ids) <= max_block:
            pairs.update(itertools.combinations(sorted(ids), 2))
    return sorted(pairs)


def pair_features(d, pairs):
    a = d.iloc[[p[0] for p in pairs]].reset_index(drop=True)
    b = d.iloc[[p[1] for p in pairs]].reset_index(drop=True)
    out = pd.DataFrame(index=range(len(pairs)))
    out["jw_first"] = [JaroWinkler.similarity(x, y) for x, y in zip(a.f, b.f)]
    out["jw_last"] = [JaroWinkler.similarity(x, y) for x, y in zip(a.l, b.l)]
    out["nick_eq"] = (a.fc == b.fc).astype(float)
    out["first_initial_eq"] = (a.f.str[:1] == b.f.str[:1]).astype(float)
    both_ad = (a.ad != "") & (b.ad != "")
    out["addr_sim"] = np.where(both_ad, [fuzz.token_set_ratio(x, y) / 100 for x, y in zip(a.ad, b.ad)], np.nan)
    out["zip_eq"] = (a.zip == b.zip).astype(float)
    out["city_eq"] = (a.city == b.city).astype(float)
    both_em = (a.em != "") & (b.em != "")
    out["email_exact"] = np.where(both_em, (a.em == b.em).astype(float), np.nan)
    out["email_local_sim"] = np.where(both_em, [JaroWinkler.similarity(x, y) for x, y in zip(a.em_local, b.em_local)], np.nan)
    both_ph = (a.ph != "") & (b.ph != "")
    out["phone_exact"] = np.where(both_ph, (a.ph == b.ph).astype(float), np.nan)
    out["email_missing"] = (~both_em).astype(float)
    out["phone_missing"] = (~both_ph).astype(float)
    out["same_source"] = (a.source == b.source).astype(float)
    out["rule_match"] = ((out.email_exact == 1) | (out.phone_exact == 1) |
                         ((a.f == b.f) & (a.l == b.l) & (a.zip == b.zip))).astype(int)
    out["label"] = (a.true_id.values == b.true_id.values).astype(int)
    out["i"] = [p[0] for p in pairs]
    out["j"] = [p[1] for p in pairs]
    out["entity_i"] = a.true_id.values
    out["entity_j"] = b.true_id.values
    return out


def _count_true_pairs(df):
    c = df.groupby("true_id").size()
    return int((c * (c - 1) // 2).sum())


def _components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return len({find(x) for x in range(n)})


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def run_matching(df, seed=7):
    """Train on one half of the entities, evaluate on held-out entities (no leakage)."""
    d = _prep(df)
    pairs = candidate_pairs(d)
    pf = pair_features(d, pairs)
    rng = np.random.default_rng(seed)
    ents = d.true_id.unique()
    test_ents = set(rng.choice(ents, size=len(ents) // 2, replace=False))
    in_test_i = pf.entity_i.isin(test_ents)
    in_test_j = pf.entity_j.isin(test_ents)
    train = pf[~in_test_i & ~in_test_j]
    test = pf[in_test_i & in_test_j].copy()
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=150, random_state=seed)
    model.fit(train[FEATURES], train.label)
    test["proba"] = model.predict_proba(test[FEATURES])[:, 1]
    test_df = d[d.true_id.isin(test_ents)]
    true_pairs = _count_true_pairs(test_df)
    importances = _perm_importance(model, test)
    return dict(model=model, test=test, test_records=len(test_df), true_pairs=true_pairs,
                true_identities=test_df.true_id.nunique(), d=d, test_idx=test_df.index.values,
                importances=importances, n_candidates=len(pf))


def _perm_importance(model, test, n=1500):
    s = test.sample(min(n, len(test)), random_state=1)
    base = (model.predict_proba(s[FEATURES])[:, 1] > 0.5).astype(int)
    rng = np.random.default_rng(0)
    res = {}
    for f in FEATURES:
        x = s[FEATURES].copy()
        x[f] = rng.permutation(x[f].values)
        pred = (model.predict_proba(x)[:, 1] > 0.5).astype(int)
        res[f] = float((pred != base).mean())
    return pd.Series(res).sort_values(ascending=False)


def evaluate(res, threshold=0.5):
    t, true_pairs = res["test"], res["true_pairs"]
    out = {}
    for name, pred in [("Rules only", t.rule_match.values), ("ML model", (t.proba >= threshold).astype(int).values)]:
        tp = int(((pred == 1) & (t.label == 1)).sum())
        fp = int(((pred == 1) & (t.label == 0)).sum())
        fn = true_pairs - tp
        p, r, f = prf(tp, fp, fn)
        edges = list(zip(t.i[pred == 1], t.j[pred == 1]))
        idx = {v: k for k, v in enumerate(res["test_idx"])}
        e2 = [(idx[i], idx[j]) for i, j in edges if i in idx and j in idx]
        out[name] = dict(precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn,
                         identities=_components(len(idx), e2))
    return out


# ---------- anomaly detection ----------
def source_batch_metrics(df):
    g = df.copy()
    g["phone_digits"] = g.phone.fillna("").str.replace(r"\D", "", regex=True)
    g["phone_bad"] = g.phone.notna() & ~g.phone_digits.str.len().isin([10, 11])
    g["stale"] = (REF_DATE - g.updated_at).dt.days > 365
    m = g.groupby(["source", "batch"]).agg(
        records=("record_id", "count"),
        email_null=("email", lambda s: s.isna().mean()),
        phone_null=("phone", lambda s: s.isna().mean()),
        phone_malformed=("phone_bad", "mean"),
        address_null=("address", lambda s: s.isna().mean()),
        consent_null=("consent", lambda s: s.isna().mean()),
        stale=("stale", "mean"),
    ).reset_index()
    return m


METRIC_COLS = ["email_null", "phone_null", "phone_malformed", "address_null", "consent_null", "stale"]


def detect_anomalies(m, contamination=0.08, seed=3):
    X = StandardScaler().fit_transform(m[METRIC_COLS])
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(X)
    out = m.copy()
    out["anomaly_score"] = -iso.score_samples(X)
    out["flagged"] = iso.predict(X) == -1
    z = pd.DataFrame(X, columns=METRIC_COLS)
    out["top_driver"] = z.abs().idxmax(axis=1)
    out["driver_z"] = z.abs().max(axis=1)
    return out


# ---------- AI-readiness score (transparent weighted score, not ML) ----------
WEIGHTS = {"identity_coverage": 0.30, "consent_coverage": 0.25, "freshness": 0.20, "attribute_depth": 0.25}


def readiness(df):
    def score(g):
        has_key = (g.email.notna() | g.phone.notna()).mean()
        consent = g.consent.notna().mean()
        fresh = ((REF_DATE - g.updated_at).dt.days <= 365).mean()
        depth = g[["first_name", "last_name", "email", "phone", "address", "consent"]].notna().mean(axis=1).mean()
        parts = dict(identity_coverage=has_key, consent_coverage=consent, freshness=fresh, attribute_depth=depth)
        parts["score"] = 100 * sum(WEIGHTS[k] * v for k, v in parts.items())
        return pd.Series(parts)
    by_src = df.groupby("source").apply(score, include_groups=False)
    overall = score(df)
    return overall, by_src


def rating(s):
    return "Ready" if s >= 80 else "Nearly ready" if s >= 65 else "Needs work"


# ---------- insights ----------
def findings(df, ev, overall, by_src, anomalies):
    flagged = anomalies[anomalies.flagged].sort_values("anomaly_score", ascending=False).head(5)
    return dict(
        records=len(df),
        readiness=round(float(overall.score), 1),
        weakest_source=by_src.score.idxmin(),
        weakest_score=round(float(by_src.score.min()), 1),
        rules=ev["Rules only"], ml=ev["ML model"],
        flagged=[f"{r.source} {r.batch}: {r.top_driver} (z={r.driver_z:.1f})" for r in flagged.itertuples()],
    )


def template_insights(f):
    r, m = f["rules"], f["ml"]
    lines = [
        f"**Readiness:** overall score {f['readiness']}/100 ({rating(f['readiness'])}). Weakest source: {f['weakest_source']} ({f['weakest_score']}).",
        f"**Matching:** on held-out data the ML model reached recall {m['recall']:.0%} at precision {m['precision']:.0%}, "
        f"versus recall {r['recall']:.0%} at precision {r['precision']:.0%} for rules only.",
        "**Data quality incidents flagged:** " + ("; ".join(f["flagged"]) if f["flagged"] else "none."),
        "**Recommended first steps:** (1) fix the flagged source/batch loads at the point of ingestion, "
        "(2) backfill consent for the lowest-scoring source, (3) route low-confidence match pairs to review.",
    ]
    return "\n\n".join(lines)


def llm_insights(f, api_key=None):
    """Optional: ask Claude to write the narrative. Falls back to the template on any problem."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return template_insights(f), False
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        prompt = (
            "You are a product manager reviewing an identity and data health report for a brand. "
            "Using ONLY the numbers below, write a short brief: what is working, the 3 biggest risks to AI-driven "
            "marketing, and 3 quick fixes ordered by impact vs effort. Do not invent numbers.\n\n" + str(f))
        msg = client.messages.create(model="claude-sonnet-5-5", max_tokens=700,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text, True
    except Exception as e:  # noqa: BLE001
        return template_insights(f) + f"\n\n_(AI narrative unavailable: {type(e).__name__}. Showing the template summary.)_", False
