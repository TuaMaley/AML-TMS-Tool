"""
AML-TMS Sanctions Screening Module
=====================================
Screens entities against OFAC SDN-equivalent list.
Uses fuzzy matching (Levenshtein distance) for name variations,
AKA matching, and partial entity name detection.

In production this would pull from:
  - OFAC SDN list (https://ofac.treasury.gov/sdn-list)
  - UN Security Council Consolidated List
  - EU Consolidated Financial Sanctions List
  - HM Treasury Financial Sanctions

For this PoC we use a representative seed dataset that mirrors
real OFAC SDN structure with realistic sanctioned entity profiles.
"""
import re, math
from datetime import datetime

# ── Representative OFAC-style SDN seed data ──────────────────────────────────
SDN_LIST = [
    # Individuals
    {"id":"SDN-001","name":"Viktor Petrov","type":"individual","program":["UKRAINE-EO13685","RUSSIA-EO14024"],
     "aka":["Victor Petrov","V. Petrov","Viktor P."],"country":"Russia","dob":"1972-03-15",
     "reason":"Designated for facilitating illicit financial transfers for sanctioned Russian entities.",
     "date_added":"2022-02-25","score_threshold":85},
    {"id":"SDN-002","name":"Ahmad Al-Rashidi","type":"individual","program":["SDGT","IRAQ2"],
     "aka":["Ahmed Al Rashidi","A. Rashidi","Ahmad Rashidi"],"country":"Iraq","dob":"1968-07-20",
     "reason":"Designated for providing financial support to terrorist organisations.",
     "date_added":"2019-06-10","score_threshold":85},
    {"id":"SDN-003","name":"Chen Wei Financial Group","type":"entity","program":["TCPA","DPRK3"],
     "aka":["CW Financial","Chen Wei Group","CWFG"],"country":"China","dob":None,
     "reason":"Front company for DPRK weapons proliferation financing.",
     "date_added":"2021-09-01","score_threshold":80},
    {"id":"SDN-004","name":"Nexus Trading LLC","type":"entity","program":["UKRAINE-EO13685"],
     "aka":["Nexus Trade","Nexus LLC","NT LLC"],"country":"Panama","dob":None,
     "reason":"Shell company used for layering sanctioned funds through U.S. financial system.",
     "date_added":"2023-01-15","score_threshold":80},
    {"id":"SDN-005","name":"Ibrahim Hassan Al-Farsi","type":"individual","program":["SDGT"],
     "aka":["I. Al-Farsi","Hassan Al Farsi","Ibrahim Alfarsi"],"country":"UAE","dob":"1975-11-08",
     "reason":"Financing network for designated terrorist organisation.",
     "date_added":"2020-04-22","score_threshold":85},
    {"id":"SDN-006","name":"GoldPath International","type":"entity","program":["IRAN","IFSR"],
     "aka":["Gold Path Intl","GoldPath Corp","GP International"],"country":"Iran","dob":None,
     "reason":"Facilitating sanctions evasion for Iranian petroleum exports.",
     "date_added":"2022-08-30","score_threshold":80},
    {"id":"SDN-007","name":"Meridian Capital Holdings","type":"entity","program":["VENEZUELA-EO13850"],
     "aka":["Meridian Holdings","MCH","Meridian Capital"],"country":"Venezuela","dob":None,
     "reason":"Designated for corruption linked to Venezuelan state oil company.",
     "date_added":"2019-11-05","score_threshold":80},
    {"id":"SDN-008","name":"Dmitri Volkov","type":"individual","program":["RUSSIA-EO14024","CAATSA"],
     "aka":["D. Volkov","Dmitry Volkov","Dmitri V."],"country":"Russia","dob":"1965-04-30",
     "reason":"Oligarch with significant Russian state energy sector interests post-invasion.",
     "date_added":"2022-04-06","score_threshold":85},
    {"id":"SDN-009","name":"Atlantic Shell Holdings","type":"entity","program":["UKRAINE-EO13685"],
     "aka":["Atlantic Shell","ASH Corp","Atlantic Holdings"],"country":"BVI","dob":None,
     "reason":"Shell company network used to obscure sanctioned Russian assets.",
     "date_added":"2023-03-10","score_threshold":80},
    {"id":"SDN-010","name":"Kim Sung Ho","type":"individual","program":["DPRK3","DPRK4"],
     "aka":["Sung Ho Kim","K. Sung Ho","S.H. Kim"],"country":"DPRK","dob":"1970-08-12",
     "reason":"Procuring technology and financing for DPRK ballistic missile programme.",
     "date_added":"2017-09-22","score_threshold":85},
    {"id":"SDN-011","name":"Vortex Crypto Exchange","type":"entity","program":["CYBER2"],
     "aka":["Vortex Exchange","Vortex Crypto","VCX"],"country":"Unknown","dob":None,
     "reason":"Primary money laundering platform for ransomware proceeds.",
     "date_added":"2022-11-01","score_threshold":80},
    {"id":"SDN-012","name":"Clearwater Export Finance","type":"entity","program":["IRAN"],
     "aka":["Clearwater Finance","CW Export","Clearwater Exports"],"country":"Iran","dob":None,
     "reason":"Front for Iranian export credit financing in violation of JCPOA sanctions.",
     "date_added":"2021-05-14","score_threshold":80},
    {"id":"SDN-013","name":"Phoenix Remittance Services","type":"entity","program":["SDGT","SOMALIA"],
     "aka":["Phoenix Remittance","PRS","Phoenix Services"],"country":"Somalia","dob":None,
     "reason":"Hawala network channelling funds to Al-Shabaab affiliate.",
     "date_added":"2020-07-19","score_threshold":80},
    {"id":"SDN-014","name":"Kwame Asante Boateng","type":"individual","program":["GLOMAG"],
     "aka":["K.A. Boateng","Kwame Boateng","K. Boateng"],"country":"Ghana","dob":"1968-02-14",
     "reason":"Corrupt official designated under Global Magnitsky Act for bribery.",
     "date_added":"2021-12-09","score_threshold":85},
    {"id":"SDN-015","name":"Harbor Digital Finance","type":"entity","program":["CYBER2","DPRK3"],
     "aka":["Harbor Digital","HDF","Harbor Finance"],"country":"Unknown","dob":None,
     "reason":"Virtual asset exchange facilitating DPRK cyber theft laundering.",
     "date_added":"2023-06-15","score_threshold":80},
]

# ── High-risk jurisdictions (FATF grey/black list + OFAC focus countries) ───
HIGH_RISK_JURISDICTIONS = {
    "Iran": {"risk": "critical", "programs": ["IRAN","IFSR","ISA"], "fatf": "blacklist"},
    "DPRK": {"risk": "critical", "programs": ["DPRK1","DPRK2","DPRK3","DPRK4"], "fatf": "blacklist"},
    "North Korea": {"risk": "critical", "programs": ["DPRK1","DPRK2","DPRK3","DPRK4"], "fatf": "blacklist"},
    "Russia": {"risk": "high", "programs": ["RUSSIA-EO14024","CAATSA","UKRAINE-EO13685"], "fatf": "greylist"},
    "Venezuela": {"risk": "high", "programs": ["VENEZUELA-EO13850"], "fatf": "greylist"},
    "Cuba": {"risk": "high", "programs": ["CUBA"], "fatf": "monitored"},
    "Syria": {"risk": "critical", "programs": ["SYRIA"], "fatf": "blacklist"},
    "Myanmar": {"risk": "high", "programs": ["BURMA-EO14014"], "fatf": "greylist"},
    "Belarus": {"risk": "high", "programs": ["BELARUS-EO14038"], "fatf": "greylist"},
    "Somalia": {"risk": "high", "programs": ["SOMALIA","SDGT"], "fatf": "greylist"},
    "Yemen": {"risk": "high", "programs": ["YEMEN"], "fatf": "greylist"},
    "Sudan": {"risk": "high", "programs": ["SUDAN"], "fatf": "greylist"},
    "Libya": {"risk": "high", "programs": ["LIBYA2"], "fatf": "greylist"},
    "Iraq": {"risk": "elevated", "programs": ["IRAQ2","SDGT"], "fatf": "monitored"},
    "Panama": {"risk": "elevated", "programs": [], "fatf": "greylist"},
    "UAE": {"risk": "elevated", "programs": [], "fatf": "greylist"},
    "BVI": {"risk": "elevated", "programs": [], "fatf": "monitored"},
    "Cayman Islands": {"risk": "elevated", "programs": [], "fatf": "monitored"},
}


# ── Fuzzy matching ────────────────────────────────────────────────────────────
def _levenshtein(s1: str, s2: str) -> int:
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2: return 0
    if len(s1) < len(s2): s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]

def _similarity(a: str, b: str) -> float:
    """Return 0-100 similarity score between two strings."""
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b: return 0
    if a == b: return 100
    # Exact substring
    if a in b or b in a:
        shorter = min(len(a), len(b))
        longer  = max(len(a), len(b))
        return round(90 * shorter / longer)
    # Token overlap
    ta = set(re.split(r'\W+', a))
    tb = set(re.split(r'\W+', b))
    ta = {t for t in ta if len(t) > 1}
    tb = {t for t in tb if len(t) > 1}
    if ta and tb:
        overlap = len(ta & tb) / max(len(ta), len(tb))
        if overlap >= 0.5:
            return round(70 + overlap * 25)
    # Edit distance
    dist = _levenshtein(a, b)
    max_len = max(len(a), len(b))
    return round(max(0, (1 - dist / max_len) * 100))

def _normalize(name: str) -> str:
    """Normalize name for comparison."""
    name = name.lower().strip()
    name = re.sub(r'[,\.\-_&]', ' ', name)
    name = re.sub(r'\b(llc|ltd|inc|corp|co|group|holdings|international|'
                  r'services|finance|capital|trading|enterprise|enterprises)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


# ── Main screening function ───────────────────────────────────────────────────
def screen_entity(name: str, country: str = None, entity_type: str = None) -> dict:
    """
    Screen an entity name against the SDN list.
    Returns matches with confidence scores and hit details.
    """
    if not name or len(name.strip()) < 2:
        return {"status": "error", "message": "Name too short to screen"}

    norm_name = _normalize(name)
    matches = []

    for _entry in SDN_LIST:
        # Normalize "program" vs "programs" key
        entry = {**_entry, "programs": _entry.get("programs", _entry.get("program", []))}
        best_score = 0
        best_match_on = "primary"

        # Score against primary name
        s = _similarity(norm_name, _normalize(entry["name"]))
        if s > best_score:
            best_score = s
            best_match_on = f"primary name: {entry['name']}"

        # Score against AKAs
        for aka in entry.get("aka", []):
            s = _similarity(norm_name, _normalize(aka))
            if s > best_score:
                best_score = s
                best_match_on = f"AKA: {aka}"

        # Boost score if country matches
        country_match = False
        if country and entry.get("country"):
            if country.lower() in entry["country"].lower() or \
               entry["country"].lower() in country.lower():
                country_match = True
                if best_score >= 60:
                    best_score = min(100, best_score + 8)

        if best_score >= 55:
            threshold = entry.get("score_threshold", 80)
            hit_type = "CONFIRMED_HIT" if best_score >= threshold else \
                       "POTENTIAL_MATCH" if best_score >= 70 else "WEAK_MATCH"
            matches.append({
                "sdn_id":       entry["id"],
                "sdn_name":     entry["name"],
                "match_score":  best_score,
                "match_on":     best_match_on,
                "hit_type":     hit_type,
                "programs":     entry["programs"],
                "country":      entry["country"],
                "entity_type":  entry["type"],
                "reason":       entry["reason"],
                "date_added":   entry["date_added"],
                "country_match":country_match,
                "dob":          entry.get("dob"),
            })

    # Sort by score descending
    matches.sort(key=lambda x: -x["match_score"])

    # Jurisdiction screening
    jurisdiction_hits = []
    if country:
        for jur_name, jur_info in HIGH_RISK_JURISDICTIONS.items():
            if country.lower() in jur_name.lower() or jur_name.lower() in country.lower():
                jurisdiction_hits.append({
                    "jurisdiction": jur_name,
                    "risk_level":   jur_info["risk"],
                    "programs":     jur_info["programs"],
                    "fatf_status":  jur_info["fatf"],
                })

    # Determine overall result
    confirmed = [m for m in matches if m["hit_type"] == "CONFIRMED_HIT"]
    potential = [m for m in matches if m["hit_type"] == "POTENTIAL_MATCH"]

    if confirmed:
        status = "HIT"
        risk_level = "critical"
    elif potential:
        status = "POTENTIAL_MATCH"
        risk_level = "high"
    elif jurisdiction_hits and jurisdiction_hits[0]["risk_level"] == "critical":
        status = "JURISDICTION_RISK"
        risk_level = "high"
    elif jurisdiction_hits:
        status = "JURISDICTION_RISK"
        risk_level = "elevated"
    elif matches:
        status = "WEAK_MATCH"
        risk_level = "medium"
    else:
        status = "CLEAR"
        risk_level = "low"

    return {
        "query":              name,
        "normalized_query":   norm_name,
        "status":             status,
        "risk_level":         risk_level,
        "matches":            matches[:5],  # top 5
        "confirmed_hits":     len(confirmed),
        "potential_matches":  len(potential),
        "jurisdiction_hits":  jurisdiction_hits,
        "screened_at":        datetime.now().isoformat(),
        "list_version":       "OFAC-SDN-2024-Q4 (representative seed)",
        "total_checked":      len(SDN_LIST),
        "action_required":    status in ("HIT", "POTENTIAL_MATCH"),
    }


def batch_screen(entities: list) -> list:
    """Screen multiple entities at once."""
    return [screen_entity(e.get("name",""), e.get("country"), e.get("type"))
            for e in entities]


def get_sdn_stats() -> dict:
    programs = {}
    countries = {}
    for e in SDN_LIST:
        for p in e.get("programs", e.get("program", [])):
            programs[p] = programs.get(p, 0) + 1
        c = e.get("country","Unknown")
        countries[c] = countries.get(c, 0) + 1
    return {
        "total_entries":    len(SDN_LIST),
        "individuals":      sum(1 for e in SDN_LIST if e["type"]=="individual"),
        "entities":         sum(1 for e in SDN_LIST if e["type"]=="entity"),
        "programs_covered": len(programs),
        "top_programs":     sorted(programs.items(), key=lambda x: -x[1])[:8],
        "countries_covered":len(countries),
        "last_updated":     "2024-Q4 (representative seed data)",
        "high_risk_jurisdictions": len(HIGH_RISK_JURISDICTIONS),
    }
