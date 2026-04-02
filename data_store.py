"""
In-memory data store for AML-TMS.
Holds alerts, cases, transactions, SAR records, audit log.
"""
import random, time, uuid
from datetime import datetime, timedelta

random.seed(99)

ENTITIES = [
    "Nexus Trading LLC", "GoldPath Remittance", "Harbor Digital Inc.",
    "Clearwater Exports", "Rivera M. (Personal)", "Meridian FX Corp",
    "Apex Holdings Ltd", "BlueStar Payments", "Orion Capital Group",
    "Delta Wire Services", "Summit Trade Finance", "Keystone MSB",
    "Phoenix Remittance", "Atlantic Shell Co.", "Vortex Crypto Ltd",
]
CHANNELS = ["Wire Transfer", "Cash Deposit", "Fintech API",
            "FX/Treasury", "Trade Finance", "Mobile Banking"]
TYPOLOGIES = [
    "Structuring / threshold evasion", "Cross-border layering",
    "Smurfing pattern", "Sanctions-adjacent activity",
    "FX structuring", "Anomalous transaction pattern",
    "Shell company network", "Crypto conversion",
]
OFFICERS = ["J. Mensah", "A. Owusu", "B. Asante", "K. Boateng", "R. Adjapong"]

def _ts(days_ago=0, hours_ago=0):
    dt = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _rand_amount():
    buckets = [
        (0.35, lambda: round(random.uniform(5000, 11000), 2)),
        (0.30, lambda: round(random.uniform(20000, 200000), 2)),
        (0.20, lambda: round(random.uniform(200000, 1500000), 2)),
        (0.15, lambda: round(random.uniform(500, 5000), 2)),
    ]
    r = random.random()
    cumulative = 0
    for prob, gen in buckets:
        cumulative += prob
        if r <= cumulative:
            return gen()
    return 50000.0


# ── Seed alerts ─────────────────────────────────────────────────────────────
ALERTS = [
    {
        "id": "ALT-2841", "entity": "Nexus Trading LLC",
        "amount": 487500.00, "score": 94, "priority": "critical",
        "typology": "Cross-border layering", "channel": "Wire Transfer",
        "timestamp": _ts(hours_ago=1), "status": "open",
        "officer": "J. Mensah", "case_id": "CAS-0411",
        "model_scores": {"iso": 88, "xgb": 97, "gnn": 95, "lstm": 72},
        "shap": [
            {"label": "3-day velocity change", "shap": 0.38, "value": 740},
            {"label": "New counterparty flag", "shap": 0.27, "value": 1},
            {"label": "Round-dollar structuring", "shap": 0.19, "value": 1},
            {"label": "Jurisdiction risk", "shap": 0.12, "value": 3},
            {"label": "Transaction hour anomaly", "shap": -0.08, "value": 14},
        ],
        "transactions": [
            {"dir": "out", "desc": "Wire → Panama City", "amount": -487500},
            {"dir": "in",  "desc": "Wire ← Cayman Is.",  "amount": 490000},
            {"dir": "out", "desc": "Wire → Delaware LLC", "amount": -122000},
        ],
        "notes": "",
    },
    {
        "id": "ALT-2840", "entity": "GoldPath Remittance",
        "amount": 9800.00, "score": 87, "priority": "high",
        "typology": "Structuring / threshold evasion", "channel": "Cash Deposit",
        "timestamp": _ts(hours_ago=2), "status": "open",
        "officer": "A. Owusu", "case_id": None,
        "model_scores": {"iso": 62, "xgb": 91, "gnn": 55, "lstm": 93},
        "shap": [
            {"label": "Round-dollar structuring", "shap": 0.42, "value": 1},
            {"label": "7-day velocity change",   "shap": 0.31, "value": 310},
            {"label": "Account age (days)",       "shap": 0.18, "value": 22},
            {"label": "Customer risk tier",       "shap": -0.05, "value": 2},
            {"label": "Prior SAR history",        "shap": -0.12, "value": 0},
        ],
        "transactions": [
            {"dir": "in", "desc": "Cash deposit Branch #4", "amount": 9800},
            {"dir": "in", "desc": "Cash deposit Branch #7", "amount": 9750},
            {"dir": "in", "desc": "Cash deposit Branch #2", "amount": 9900},
        ],
        "notes": "",
    },
    {
        "id": "ALT-2839", "entity": "Harbor Digital Inc.",
        "amount": 234000.00, "score": 81, "priority": "high",
        "typology": "Crypto conversion", "channel": "Fintech API",
        "timestamp": _ts(hours_ago=3), "status": "review",
        "officer": "J. Mensah", "case_id": "CAS-0409",
        "model_scores": {"iso": 79, "xgb": 83, "gnn": 88, "lstm": 74},
        "shap": [
            {"label": "Multi-currency conversion", "shap": 0.36, "value": 1},
            {"label": "Cross-border transaction",  "shap": 0.29, "value": 1},
            {"label": "Amount vs peer group %",    "shap": 0.21, "value": 480},
            {"label": "Account age (days)",        "shap": 0.09, "value": 28},
            {"label": "Customer risk tier",        "shap": -0.04, "value": 1},
        ],
        "transactions": [
            {"dir": "in",  "desc": "API transfer ← Exchange", "amount": 234000},
            {"dir": "out", "desc": "Crypto conversion outflow", "amount": -230000},
        ],
        "notes": "Possible virtual asset layering. Counterparty flagged in prior period.",
    },
    {
        "id": "ALT-2838", "entity": "Clearwater Exports",
        "amount": 1200000.00, "score": 76, "priority": "high",
        "typology": "Shell company network", "channel": "Trade Finance",
        "timestamp": _ts(hours_ago=5), "status": "open",
        "officer": "B. Asante", "case_id": None,
        "model_scores": {"iso": 70, "xgb": 78, "gnn": 82, "lstm": 65},
        "shap": [
            {"label": "Transaction amount",       "shap": 0.44, "value": 1200000},
            {"label": "Jurisdiction risk",        "shap": 0.28, "value": 3},
            {"label": "Cross-border transaction", "shap": 0.16, "value": 1},
            {"label": "3-day velocity change",    "shap": 0.07, "value": 220},
            {"label": "Amount vs peer group %",   "shap": -0.18, "value": 85},
        ],
        "transactions": [
            {"dir": "out", "desc": "LC payment → Panama", "amount": -1200000},
            {"dir": "in",  "desc": "Invoice settlement",  "amount": 980000},
        ],
        "notes": "",
    },
    {
        "id": "ALT-2837", "entity": "Rivera M. (Personal)",
        "amount": 4200.00, "score": 68, "priority": "medium",
        "typology": "Smurfing pattern", "channel": "Mobile Banking",
        "timestamp": _ts(hours_ago=6), "status": "open",
        "officer": None, "case_id": None,
        "model_scores": {"iso": 55, "xgb": 72, "gnn": 48, "lstm": 81},
        "shap": [
            {"label": "7-day velocity change",     "shap": 0.39, "value": 190},
            {"label": "Round-dollar structuring",  "shap": 0.25, "value": 1},
            {"label": "Account age (days)",        "shap": 0.22, "value": 45},
            {"label": "Prior SAR history",         "shap": -0.19, "value": 0},
            {"label": "Customer risk tier",        "shap": -0.14, "value": 1},
        ],
        "transactions": [
            {"dir": "in", "desc": "Mobile deposit", "amount": 4200},
            {"dir": "in", "desc": "Mobile deposit", "amount": 4150},
            {"dir": "in", "desc": "Mobile deposit", "amount": 4300},
            {"dir": "in", "desc": "Mobile deposit", "amount": 3950},
        ],
        "notes": "",
    },
    {
        "id": "ALT-2836", "entity": "Meridian FX Corp",
        "amount": 89300.00, "score": 65, "priority": "medium",
        "typology": "FX structuring", "channel": "FX/Treasury",
        "timestamp": _ts(hours_ago=8), "status": "cleared",
        "officer": "K. Boateng", "case_id": None,
        "model_scores": {"iso": 58, "xgb": 67, "gnn": 60, "lstm": 70},
        "shap": [
            {"label": "Multi-currency conversion", "shap": 0.31, "value": 1},
            {"label": "Cross-border transaction",  "shap": 0.28, "value": 1},
            {"label": "Counterparty network degree","shap": 0.19, "value": 12},
            {"label": "Amount vs peer group %",    "shap": -0.22, "value": 92},
            {"label": "7-day velocity change",     "shap": -0.15, "value": 40},
        ],
        "transactions": [
            {"dir": "out", "desc": "FX conversion EUR→USD", "amount": -89300},
            {"dir": "in",  "desc": "FX receipt",            "amount": 88100},
        ],
        "notes": "Cleared — verified MSB license, consistent FX history.",
    },
]

# ── Seed cases ───────────────────────────────────────────────────────────────
CASES = [
    {
        "id": "CAS-0411", "entity": "Nexus Trading LLC",
        "alerts": ["ALT-2841"], "alert_count": 3, "priority": "critical",
        "status": "review", "officer": "J. Mensah",
        "opened": _ts(hours_ago=1), "sar_due": _ts(days_ago=-29),
        "typology": "Cross-border layering",
        "narrative": "Entity identified in 3 separate alert events involving cross-border wire transfers to high-risk jurisdictions. Pattern consistent with layering through shell company network. Escalated for SAR determination.",
        "sar_status": "pending",
    },
    {
        "id": "CAS-0410", "entity": "GoldPath Remittance",
        "alerts": ["ALT-2840"], "alert_count": 1, "priority": "high",
        "status": "open", "officer": "A. Owusu",
        "opened": _ts(hours_ago=2), "sar_due": _ts(days_ago=-30),
        "typology": "Structuring",
        "narrative": "Single alert on structuring pattern — multiple just-below-threshold deposits across branches. Under investigation.",
        "sar_status": "under_review",
    },
    {
        "id": "CAS-0409", "entity": "Harbor Digital Inc.",
        "alerts": ["ALT-2839"], "alert_count": 2, "priority": "high",
        "status": "review", "officer": "J. Mensah",
        "opened": _ts(hours_ago=3), "sar_due": _ts(days_ago=-30),
        "typology": "Crypto conversion",
        "narrative": "Virtual asset layering pattern. Rapid crypto conversion following API inflow. GNN model flagged counterparty network involvement.",
        "sar_status": "pending",
    },
    {
        "id": "CAS-0407", "entity": "Kwame B. (Personal)",
        "alerts": [], "alert_count": 4, "priority": "medium",
        "status": "filed", "officer": "B. Asante",
        "opened": _ts(days_ago=3), "sar_due": _ts(days_ago=-27),
        "typology": "Smurfing pattern",
        "narrative": "SAR filed with FinCEN on structuring activity across 4 alerts. BSA e-filing complete.",
        "sar_status": "filed",
    },
]

# ── Audit log ────────────────────────────────────────────────────────────────
AUDIT_LOG = [
    {"ts": _ts(hours_ago=1),  "user": "system",     "action": "ALERT_GENERATED",  "target": "ALT-2841", "detail": "ML score 94 — threshold exceeded"},
    {"ts": _ts(hours_ago=1),  "user": "J. Mensah",  "action": "ALERT_ASSIGNED",   "target": "ALT-2841", "detail": "Assigned to J. Mensah (critical queue)"},
    {"ts": _ts(hours_ago=1),  "user": "system",     "action": "CASE_CREATED",     "target": "CAS-0411", "detail": "Auto-case created from critical alert"},
    {"ts": _ts(hours_ago=2),  "user": "system",     "action": "ALERT_GENERATED",  "target": "ALT-2840", "detail": "ML score 87 — threshold exceeded"},
    {"ts": _ts(hours_ago=8),  "user": "K. Boateng", "action": "ALERT_CLEARED",    "target": "ALT-2836", "detail": "Cleared as FP — verified MSB license"},
    {"ts": _ts(days_ago=3),   "user": "B. Asante",  "action": "SAR_FILED",        "target": "CAS-0407", "detail": "FinCEN Form 111 submitted via BSA e-filing"},
]

# ── SAR records ──────────────────────────────────────────────────────────────
SAR_RECORDS = [
    {
        "id": "SAR-0051", "case_id": "CAS-0407", "entity": "Kwame B. (Personal)",
        "amount": 18400.00, "typology": "Structuring",
        "filed_date": _ts(days_ago=1), "officer": "B. Asante",
        "status": "filed", "fincen_ref": "FIN-2024-0051",
    },
]

# ── Live transaction feed generator ─────────────────────────────────────────
_feed_counter = 2842

def generate_live_transaction():
    global _feed_counter
    score = random.choices(
        [random.randint(0, 40), random.randint(41, 64),
         random.randint(65, 84), random.randint(85, 100)],
        weights=[60, 25, 12, 3]
    )[0]
    entity = random.choice(ENTITIES)
    channel = random.choice(CHANNELS)
    amount = _rand_amount()
    alert_id = None
    if score >= 65:
        alert_id = f"ALT-{_feed_counter}"
        _feed_counter += 1
    return {
        "id": str(uuid.uuid4())[:8],
        "entity": entity,
        "amount": amount,
        "channel": channel,
        "score": score,
        "alert_id": alert_id,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "action": "ALERT" if score >= 65 else ("MONITOR" if score >= 40 else "PASS"),
    }

# ── System stats ─────────────────────────────────────────────────────────────
def get_system_stats():
    open_alerts = [a for a in ALERTS if a["status"] == "open"]
    critical = [a for a in ALERTS if a["priority"] == "critical"]
    avg_score = round(sum(a["score"] for a in open_alerts) / max(len(open_alerts), 1))
    return {
        "alerts_today": 47,
        "open_alerts": len(open_alerts),
        "critical_alerts": len(critical),
        "avg_score_open": avg_score,
        "cleared_today": 31,
        "sar_filed_30d": 14,
        "sar_pending": 2,
        "avg_review_time_h": 2.1,
        "fp_reduction_pct": 40,
        "investigator_hours_saved_pct": 51,
        "txns_per_min": round(12800 + random.uniform(-200, 200)),
        "pipeline_latency_ms": round(138 + random.uniform(-15, 20)),
        "data_quality_score": round(97.1 + random.uniform(-0.3, 0.3), 1),
    }

TIERS = ["Low", "Medium", "High", "PEP"]
JURISDICTIONS = ["Standard", "Elevated", "High", "OFAC-adjacent"]

