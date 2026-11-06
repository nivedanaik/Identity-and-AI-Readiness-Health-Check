"""Synthetic, deliberately messy customer data. No real or employer data is used."""
import random
import numpy as np
import pandas as pd
from faker import Faker

NICKNAMES = {
    "Robert": ["Bob", "Rob"], "William": ["Bill", "Will"], "Elizabeth": ["Liz", "Beth"],
    "Michael": ["Mike"], "Jennifer": ["Jen", "Jenny"], "Katherine": ["Kate", "Katie"],
    "Christopher": ["Chris"], "Jessica": ["Jess"], "Matthew": ["Matt"], "Daniel": ["Dan", "Danny"],
    "Margaret": ["Maggie", "Meg"], "Richard": ["Rick", "Dick"], "Thomas": ["Tom"],
    "Patricia": ["Pat", "Patty"], "Joseph": ["Joe"], "Samantha": ["Sam"], "Andrew": ["Andy"],
    "Rebecca": ["Becky"], "Anthony": ["Tony"], "Deborah": ["Debbie"],
}
SOURCES = ["CRM", "Web", "Retail", "EmailList"]
BATCHES = [f"W{i:02d}" for i in range(1, 11)]
ABBR = {"Street": "St", "Avenue": "Ave", "Road": "Rd", "Drive": "Dr", "Lane": "Ln", "Boulevard": "Blvd", "Court": "Ct"}
REF_DATE = pd.Timestamp("2026-10-01")


def _typo(s, rng):
    if len(s) < 4:
        return s
    i = int(rng.integers(1, len(s) - 1))
    kind = rng.integers(0, 3)
    if kind == 0:
        return s[:i] + s[i + 1:]
    if kind == 1:
        return s[:i] + s[i] + s[i:]
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def _phone_fmt(d, rng):
    k = int(rng.integers(0, 4))
    if k == 0:
        return d
    if k == 1:
        return f"{d[:3]}-{d[3:6]}-{d[6:]}"
    if k == 2:
        return f"({d[:3]}) {d[3:6]}-{d[6:]}"
    return f"+1{d}"


def generate(n_entities=1500, seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    Faker.seed(seed)
    fake = Faker("en_US")
    rows = []
    rid = 0
    nick_keys = list(NICKNAMES)
    for eid in range(n_entities):
        first = random.choice(nick_keys) if rng.random() < 0.5 else fake.first_name()
        last = fake.last_name()
        domain = random.choice(["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "aol.com"])
        style = int(rng.integers(0, 3))
        local = [f"{first}.{last}", f"{first[0]}{last}", f"{first}{last}{int(rng.integers(1, 99))}"][style].lower()
        email = f"{local}@{domain}"
        phone = f"{int(rng.integers(200, 999))}{int(rng.integers(200, 999))}{int(rng.integers(0, 9999)):04d}"
        street, city, zipc = fake.street_address().split("\n")[0], fake.city(), fake.zipcode()
        n_rec = int(rng.choice([1, 2, 3, 4], p=[0.35, 0.30, 0.20, 0.15]))
        for _ in range(n_rec):
            src = str(rng.choice(SOURCES))
            batch = str(rng.choice(BATCHES))
            f, l, e, p, a, c, z = first, last, email, phone, street, city, zipc
            r = rng.random()
            if first in NICKNAMES and r < 0.30:
                f = random.choice(NICKNAMES[first])
            elif r < 0.38:
                f = _typo(f, rng)
            elif r < 0.43:
                f = f[0]
            if rng.random() < 0.06:
                l = _typo(l, rng)
            if rng.random() < 0.15:
                l = l.upper() if rng.random() < 0.5 else l.lower()
            if rng.random() < 0.25:
                e = None
            elif rng.random() < 0.15:
                e = f"{first.lower()}{int(rng.integers(1, 999))}@{random.choice(['gmail.com', 'yahoo.com'])}"
            p = None if rng.random() < 0.30 else _phone_fmt(p, rng)
            if rng.random() < 0.12:  # moved
                a, c, z = fake.street_address().split("\n")[0], fake.city(), fake.zipcode()
            if rng.random() < 0.25:
                a = None
            elif a:
                for k, v in ABBR.items():
                    if rng.random() < 0.6:
                        a = a.replace(k, v)
            consent_p = {"CRM": 0.12, "Web": 0.35, "Retail": 0.18, "EmailList": 0.10}[src]
            consent = None if rng.random() < consent_p else str(rng.choice(["Y", "N"], p=[0.78, 0.22]))
            age_days = int(min(rng.exponential(300), 1500))
            rows.append(dict(record_id=rid, source=src, batch=batch, first_name=f, last_name=l, email=e,
                             phone=p, address=a, city=c, zip=z, consent=consent,
                             updated_at=REF_DATE - pd.Timedelta(days=age_days), true_id=eid))
            rid += 1
    df = pd.DataFrame(rows)
    # Injected data-quality incidents (so the anomaly model has something real to find)
    m = (df.source == "Web") & df.batch.isin(["W07", "W08"])
    df.loc[m & (rng.random(len(df)) < 0.85), "email"] = None
    bad = m & (rng.random(len(df)) < 0.6) & df.phone.notna()
    df.loc[bad, "phone"] = df.loc[bad, "phone"].str.replace(r"\D", "", regex=True).str[:7]
    m2 = (df.source == "Retail") & (df.batch == "W04")
    df.loc[m2 & (rng.random(len(df)) < 0.9), "consent"] = None
    m3 = (df.source == "CRM") & (df.batch == "W02")
    df.loc[m3, "updated_at"] = REF_DATE - pd.Timedelta(days=900)
    return df
