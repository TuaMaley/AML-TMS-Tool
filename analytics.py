"""
AML-TMS Analytics & Reporting Module
=======================================
Generates:
  - Executive dashboard (monthly/quarterly KPIs)
  - Trend analysis (typology trends, geographic risk)
  - PDF report data (structured for frontend PDF generation)
  - Peer benchmarking
  - Geographic risk heatmap data
"""
import random, math
from datetime import datetime, timedelta
from collections import defaultdict

random.seed(42)

TYPOLOGIES = [
    "Structuring", "Layering", "Sanctions", "Crypto/Virtual Assets",
    "Trade-Based AML", "Smurfing", "Shell Company", "Fraud/Cybercrime"
]

CHANNELS = ["Wire Transfer", "Cash Deposit", "Fintech API",
            "FX/Treasury", "Trade Finance", "Mobile Banking"]

REGIONS = [
    "North America", "Western Europe", "Eastern Europe",
    "Middle East", "Asia Pacific", "Latin America",
    "Sub-Saharan Africa", "Caribbean/Offshore"
]

RISK_COUNTRIES = {
    "United States": 2,  "United Kingdom": 2, "Germany": 2,
    "Panama": 7,         "Cayman Islands": 7, "BVI": 7,
    "Russia": 9,         "Iran": 10,          "North Korea": 10,
    "UAE": 6,            "China": 5,          "Nigeria": 6,
    "Switzerland": 4,    "Luxembourg": 4,     "Netherlands": 3,
    "Cyprus": 7,         "Malta": 6,          "Belize": 7,
    "Venezuela": 8,      "Syria": 10,         "Belarus": 8,
    "Myanmar": 8,        "Somalia": 9,        "Yemen": 9,
}


def _months_back(n: int) -> list:
    months = []
    now = datetime.now()
    for i in range(n - 1, -1, -1):
        d = now - timedelta(days=i * 30)
        months.append(d.strftime("%b %Y"))
    return months


def get_executive_dashboard(period: str = "quarterly") -> dict:
    """
    Generate executive KPI dashboard data.
    period: 'monthly' (12 months) or 'quarterly' (8 quarters)
    """
    n = 12 if period == "monthly" else 8
    labels = _months_back(n) if period == "monthly" else [f"Q{((i)%4)+1} {2024+(i//4)}" for i in range(n)]

    def trend(base, vol=0.08, drift=-0.02):
        vals = []
        v = base
        for _ in range(n):
            v = max(0, v * (1 + random.uniform(-vol, vol) + drift))
            vals.append(round(v, 1))
        return vals

    # Alert metrics
    alert_volume  = [int(random.gauss(145, 12)) for _ in range(n)]
    alert_volume  = [max(80, v) for v in alert_volume]
    fp_rates      = [round(random.gauss(58, 3), 1) for _ in range(n)]  # % after ML
    sar_count     = [int(random.gauss(14, 3)) for _ in range(n)]
    sar_count     = [max(6, s) for s in sar_count]
    review_hours  = [round(random.gauss(2.1, 0.3), 1) for _ in range(n)]

    # Regulatory
    exam_score    = [round(random.gauss(94, 2), 1) for _ in range(n)]
    sar_on_time   = [round(random.gauss(98.5, 1), 1) for _ in range(n)]
    fincen_quality= [round(random.gauss(88, 4), 1) for _ in range(n)]

    # Financial impact
    txn_volume_b  = [round(random.gauss(4.2, 0.3), 2) for _ in range(n)]
    suspicious_m  = [round(random.gauss(12.4, 2.1), 1) for _ in range(n)]
    enforcement_  = [round(random.gauss(0, 0.5), 1) for _ in range(n)]
    enforcement_  = [abs(v) for v in enforcement_]

    # Typology breakdown (current period)
    typo_counts = {t: random.randint(2, 28) for t in TYPOLOGIES}
    channel_counts = {c: random.randint(5, 40) for c in CHANNELS}

    # KPI summaries (current vs prior period)
    def delta(vals, positive_is_good=True):
        if len(vals) < 2: return 0
        cur, prev = vals[-1], vals[-2]
        chg = round(((cur - prev) / max(abs(prev), 0.01)) * 100, 1)
        return chg

    return {
        "period":   period,
        "labels":   labels,
        "generated_at": datetime.now().isoformat(),

        "kpis": {
            "total_alerts":     {"value": sum(alert_volume[-3:]), "delta": delta(alert_volume, False),  "label": "Alerts (last 3 periods)", "unit": ""},
            "fp_rate":          {"value": round(sum(fp_rates[-3:])/3, 1), "delta": delta(fp_rates, False), "label": "Avg FP rate", "unit": "%"},
            "sars_filed":       {"value": sum(sar_count[-3:]),   "delta": delta(sar_count),             "label": "SARs filed (last 3 periods)", "unit": ""},
            "avg_review_time":  {"value": round(sum(review_hours[-3:])/3, 1), "delta": delta(review_hours, False), "label": "Avg review time", "unit": "h"},
            "exam_score":       {"value": round(sum(exam_score[-3:])/3, 1),   "delta": delta(exam_score),  "label": "Avg exam score", "unit": "%"},
            "sar_on_time":      {"value": round(sum(sar_on_time[-3:])/3, 1),  "delta": delta(sar_on_time), "label": "SAR on-time rate", "unit": "%"},
            "txn_volume":       {"value": round(sum(txn_volume_b[-3:]), 1),   "delta": delta(txn_volume_b),"label": "Txn volume (last 3 periods)", "unit": "B"},
            "suspicious_value": {"value": round(sum(suspicious_m[-3:]), 1),  "delta": delta(suspicious_m),"label": "Suspicious activity value", "unit": "M"},
        },

        "series": {
            "alert_volume":   alert_volume,
            "fp_rates":       fp_rates,
            "sar_count":      sar_count,
            "review_hours":   review_hours,
            "exam_score":     exam_score,
            "sar_on_time":    sar_on_time,
            "txn_volume_b":   txn_volume_b,
            "suspicious_m":   suspicious_m,
        },

        "breakdown": {
            "by_typology": typo_counts,
            "by_channel":  channel_counts,
        },

        "benchmarks": {
            "industry_fp_rate":    62.4,
            "industry_review_time":3.8,
            "industry_sar_rate":   2.1,
            "our_fp_rate":         round(sum(fp_rates[-3:])/3, 1),
            "our_review_time":     round(sum(review_hours[-3:])/3, 1),
            "our_sar_rate":        round(sum(sar_count[-3:]) / max(sum(alert_volume[-3:]),1) * 100, 2),
        }
    }


def get_trend_analysis(months: int = 12) -> dict:
    """
    Generate typology trend data over time.
    """
    labels = _months_back(months)

    # Per-typology trends with different growth/decline patterns
    typology_trends = {}
    for i, t in enumerate(TYPOLOGIES):
        base = random.randint(8, 25)
        drift = random.choice([-0.03, -0.01, 0.0, 0.01, 0.02, 0.04])
        series = []
        v = base
        for _ in range(months):
            v = max(0, v + v * drift + random.gauss(0, 1.5))
            series.append(round(v, 1))
        typology_trends[t] = series

    # Emerging typologies (fastest growing last 3 months)
    emerging = []
    for t, vals in typology_trends.items():
        if len(vals) >= 4:
            recent_avg = sum(vals[-3:]) / 3
            prev_avg   = sum(vals[-6:-3]) / 3 if len(vals) >= 6 else vals[0]
            growth = ((recent_avg - prev_avg) / max(prev_avg, 1)) * 100
            if growth > 5:
                emerging.append({"typology": t, "growth_pct": round(growth, 1),
                                  "recent_avg": round(recent_avg, 1)})
    emerging.sort(key=lambda x: -x["growth_pct"])

    # Channel mix over time
    channel_trends = {}
    for c in CHANNELS:
        base = random.randint(5, 35)
        channel_trends[c] = [max(0, int(random.gauss(base, base*0.15))) for _ in range(months)]

    # Alert score distribution over time (has it improved?)
    score_bands = {
        "critical (85-100)": [random.randint(3, 8)  for _ in range(months)],
        "high (70-84)":      [random.randint(8, 18) for _ in range(months)],
        "medium (55-69)":    [random.randint(12,25) for _ in range(months)],
        "low (<55)":         [random.randint(20,45) for _ in range(months)],
    }

    # SAR outcome trend
    sar_trends = {
        "filed":   [random.randint(10, 18) for _ in range(months)],
        "declined":[random.randint(3, 9)   for _ in range(months)],
        "pending": [random.randint(1, 4)   for _ in range(months)],
    }

    return {
        "labels":          labels,
        "typology_trends": typology_trends,
        "emerging":        emerging[:5],
        "channel_trends":  channel_trends,
        "score_bands":     score_bands,
        "sar_trends":      sar_trends,
        "generated_at":    datetime.now().isoformat(),
    }


def get_geographic_risk() -> dict:
    """
    Generate geographic risk heatmap data.
    Returns country-level risk scores and transaction volumes.
    """
    countries = []
    for country, base_risk in RISK_COUNTRIES.items():
        noise = random.uniform(-0.5, 0.5)
        risk  = min(10, max(1, round(base_risk + noise, 1)))
        vol   = int(random.gauss(50, 20) * (11 - risk))
        countries.append({
            "country":      country,
            "risk_score":   risk,
            "risk_level":   "critical" if risk >= 9 else "high" if risk >= 7 else
                            "elevated" if risk >= 5 else "standard",
            "txn_volume":   max(1, vol),
            "alert_count":  int(vol * risk * 0.02),
            "sar_count":    int(vol * risk * 0.002),
            "flag":         _flag(country),
        })

    countries.sort(key=lambda x: -x["risk_score"])

    # Region rollup
    region_risk = {r: round(random.uniform(2, 9), 1) for r in REGIONS}
    region_risk["Eastern Europe"] = round(random.uniform(7, 9), 1)
    region_risk["Middle East"]    = round(random.uniform(6, 8), 1)
    region_risk["Caribbean/Offshore"] = round(random.uniform(6, 8), 1)

    return {
        "countries":   countries,
        "region_risk": region_risk,
        "top_risk":    countries[:8],
        "generated_at":datetime.now().isoformat(),
    }


def get_pdf_report_data(report_type: str = "quarterly",
                        period_label: str = None) -> dict:
    """
    Assemble all data needed for a regulatory PDF report.
    """
    period_label = period_label or f"Q{(datetime.now().month-1)//3+1} {datetime.now().year}"
    exec_data  = get_executive_dashboard("quarterly")
    trend_data = get_trend_analysis(12)
    geo_data   = get_geographic_risk()
    kpis = exec_data["kpis"]

    return {
        "report_type":  report_type,
        "period":       period_label,
        "institution":  "First National Compliance Bank",
        "prepared_by":  "AML Compliance Division",
        "prepared_at":  datetime.now().strftime("%B %d, %Y"),
        "generated_at": datetime.now().isoformat(),
        "confidential": True,

        "executive_summary": {
            "total_alerts":   kpis["total_alerts"]["value"],
            "fp_rate":        kpis["fp_rate"]["value"],
            "sars_filed":     kpis["sars_filed"]["value"],
            "review_time":    kpis["avg_review_time"]["value"],
            "exam_score":     kpis["exam_score"]["value"],
            "sar_on_time":    kpis["sar_on_time"]["value"],
            "txn_volume_b":   kpis["txn_volume"]["value"],
            "narrative":      f"""During {period_label}, the institution processed approximately """
                              f"""${kpis['txn_volume']['value']}B in total transaction volume through """
                              f"""its AI/ML Transaction Monitoring System. The system generated """
                              f"""{kpis['total_alerts']['value']} alerts, of which """
                              f"""{kpis['sars_filed']['value']} resulted in SAR filings to FinCEN. """
                              f"""The false-positive rate of {kpis['fp_rate']['value']}% represents a """
                              f"""40% improvement over the prior rule-based baseline of 98%. """
                              f"""All {kpis['sars_filed']['value']} SARs were filed within the BSA 30-day """
                              f"""deadline ({kpis['sar_on_time']['value']}% on-time rate). The institution """
                              f"""achieved a {kpis['exam_score']['value']}% score on its most recent """
                              f"""AML regulatory examination, with no material findings on transaction monitoring.""",
        },

        "typology_breakdown": exec_data["breakdown"]["by_typology"],
        "channel_breakdown":  exec_data["breakdown"]["by_channel"],
        "emerging_typologies":trend_data["emerging"],
        "top_risk_countries": geo_data["top_risk"][:6],

        "benchmarks":  exec_data["benchmarks"],
        "series_data": exec_data["series"],
        "labels":      exec_data["labels"],

        "regulatory_statement": (
            f"This report is prepared pursuant to the Bank Secrecy Act (31 U.S.C. §5318) "
            f"and the Anti-Money Laundering Act of 2020 (Pub. L. 116-283). All SARs referenced "
            f"herein were filed with FinCEN via BSA E-Filing in compliance with 31 C.F.R. §1020.320. "
            f"This report is confidential and intended solely for regulatory and internal compliance use."
        ),
    }


def _flag(country: str) -> str:
    flags = {
        "United States":"🇺🇸","United Kingdom":"🇬🇧","Germany":"🇩🇪",
        "Panama":"🇵🇦","Cayman Islands":"🇰🇾","BVI":"🇻🇬",
        "Russia":"🇷🇺","Iran":"🇮🇷","North Korea":"🇰🇵",
        "UAE":"🇦🇪","China":"🇨🇳","Nigeria":"🇳🇬",
        "Switzerland":"🇨🇭","Luxembourg":"🇱🇺","Netherlands":"🇳🇱",
        "Cyprus":"🇨🇾","Malta":"🇲🇹","Belize":"🇧🇿",
        "Venezuela":"🇻🇪","Syria":"🇸🇾","Belarus":"🇧🇾",
        "Myanmar":"🇲🇲","Somalia":"🇸🇴","Yemen":"🇾🇪",
    }
    return flags.get(country, "🌐")
