"""
AML-TMS REST API Server
Uses Python's built-in http.server — no external dependencies.
All endpoints return JSON. CORS enabled for local frontend.

Endpoints:
  GET  /api/health
  GET  /api/stats
  GET  /api/alerts
  GET  /api/alerts/{id}
  POST /api/alerts/{id}/action      body: {"action":"clear"|"create_case","notes":"..."}
  GET  /api/cases
  GET  /api/cases/{id}
  POST /api/cases/{id}/update       body: {"status":..., "narrative":..., "officer":...}
  POST /api/score                   body: transaction dict
  GET  /api/models/metrics
  GET  /api/models/importances
  GET  /api/models/history
  POST /api/models/retrain
  GET  /api/pipeline/stats
  GET  /api/live/transaction        returns one synthetic live transaction
  GET  /api/audit
  GET  /api/sar
"""
import json, sys, os, time, threading, uuid, random
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from ml_engine import get_engine, FEATURE_COLS
from data_store import (
    ALERTS, CASES, AUDIT_LOG, SAR_RECORDS,
    get_system_stats, generate_live_transaction,
    CHANNELS, TYPOLOGIES, TIERS, JURISDICTIONS,
)

PORT = 8787

def _log(entry):
    AUDIT_LOG.insert(0, {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **entry
    })

class AMLHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def _send(self, data, status=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        try:
            with open(path, "rb") as f:
                content = f.read()
            ct = "text/html" if path.endswith(".html") else \
                 "application/javascript" if path.endswith(".js") else \
                 "text/css" if path.endswith(".css") else "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = [p for p in path.split("/") if p]

        # ── Serve frontend ──────────────────────────────────────────────────
        if path == "" or path == "/" or path == "/index.html":
            fe = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
            self._send_file(fe)
            return

        # ── API routes ───────────────────────────────────────────────────────
        if parts[0] != "api":
            # SPA fallback — serve index.html for any non-API route
            fe = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
            self._send_file(fe)
            return

        route = "/".join(parts[1:])

        if route == "health":
            self._send({"status": "ok", "ts": datetime.now().isoformat()})

        elif route == "stats":
            self._send(get_system_stats())

        elif route == "alerts":
            status_filter = parse_qs(parsed.query).get("status", [None])[0]
            alerts = ALERTS if not status_filter else [a for a in ALERTS if a["status"] == status_filter]
            self._send({"alerts": alerts, "total": len(alerts)})

        elif len(parts) == 3 and parts[1] == "alerts":
            alert_id = parts[2]
            alert = next((a for a in ALERTS if a["id"] == alert_id), None)
            if alert:
                self._send(alert)
            else:
                self._send({"error": "Not found"}, 404)

        elif route == "cases":
            self._send({"cases": CASES, "total": len(CASES)})

        elif len(parts) == 3 and parts[1] == "cases":
            case_id = parts[2]
            case = next((c for c in CASES if c["id"] == case_id), None)
            if case:
                # Enrich with alerts
                case_alerts = [a for a in ALERTS if a.get("case_id") == case_id]
                self._send({**case, "alert_details": case_alerts})
            else:
                self._send({"error": "Not found"}, 404)

        elif route == "models/metrics":
            engine = get_engine()
            metrics = engine.get_metrics()
            importances = engine.get_feature_importances()
            self._send({
                "metrics": metrics,
                "trained_at": engine.trained_at,
                "feature_importances": importances,
            })

        elif route == "models/importances":
            engine = get_engine()
            imp = engine.get_feature_importances()
            from ml_engine import SHAP_LABELS
            result = [
                {"feature": k, "label": SHAP_LABELS.get(k, k), "importance": round(v, 4)}
                for k, v in sorted(imp.items(), key=lambda x: -x[1])
            ]
            self._send(result)

        elif route == "models/history":
            self._send(get_engine().get_training_history())

        elif route == "pipeline/stats":
            stats = get_system_stats()
            self._send({
                "txns_per_min": stats["txns_per_min"],
                "latency_ms": stats["pipeline_latency_ms"],
                "data_quality": stats["data_quality_score"],
                "features_count": 214,
                "feature_categories": {
                    "Velocity & Volume": 52,
                    "Network / Graph": 48,
                    "Behavioural Baseline": 44,
                    "Typology Indicators": 42,
                    "Entity Risk Context": 28,
                },
                "channels_live": CHANNELS,
            })

        elif route == "live/transaction":
            self._send(generate_live_transaction())

        elif route == "audit":
            limit = int(parse_qs(parsed.query).get("limit", [50])[0])
            self._send({"log": AUDIT_LOG[:limit]})

        elif route == "sar":
            self._send({"records": SAR_RECORDS, "total": len(SAR_RECORDS)})

        else:
            self._send({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = [p for p in path.split("/") if p]

        if not parts or parts[0] != "api":
            self.send_response(404); self.end_headers(); return

        route = "/".join(parts[1:])
        body = self._body()

        # ── Score transaction ────────────────────────────────────────────────
        if route == "score":
            engine = get_engine()
            # Map API fields to engine feature names
            txn = {
                "amount":             float(body.get("amount", 50000)),
                "channel_idx":        CHANNELS.index(body.get("channel", "Wire Transfer")) if body.get("channel") in CHANNELS else 0,
                "tier_idx":           TIERS.index(body.get("tier", "Medium")) if body.get("tier") in TIERS else 1,
                "velocity_3d":        float(body.get("velocity_3d", 30)),
                "velocity_7d":        float(body.get("velocity_7d", 40)),
                "new_counterparty":   int(body.get("new_counterparty", 0)),
                "jurisdiction_idx":   JURISDICTIONS.index(body.get("jurisdiction", "Standard")) if body.get("jurisdiction") in JURISDICTIONS else 0,
                "round_dollar":       int(body.get("round_dollar", 0)),
                "hour_of_day":        int(body.get("hour_of_day", 12)),
                "counterparty_degree":int(body.get("counterparty_degree", 3)),
                "cross_border":       int(body.get("cross_border", 0)),
                "account_age_days":   int(body.get("account_age_days", 365)),
                "prior_sars":         int(body.get("prior_sars", 0)),
                "amount_vs_peer_pct": float(body.get("amount_vs_peer_pct", 100)),
                "multi_currency":     int(body.get("multi_currency", 0)),
            }
            result = engine.score_transaction(txn)
            _log({"user": "api", "action": "TRANSACTION_SCORED",
                  "target": "-", "detail": f"Score: {result['score']} / {result['priority']}"})
            self._send(result)

        # ── Alert actions ────────────────────────────────────────────────────
        elif len(parts) == 4 and parts[1] == "alerts" and parts[3] == "action":
            alert_id = parts[2]
            alert = next((a for a in ALERTS if a["id"] == alert_id), None)
            if not alert:
                self._send({"error": "Alert not found"}, 404); return
            action = body.get("action")
            notes  = body.get("notes", "")
            officer = body.get("officer", "Unknown")
            if action == "clear":
                alert["status"] = "cleared"
                alert["notes"] = notes
                _log({"user": officer, "action": "ALERT_CLEARED",
                      "target": alert_id, "detail": notes or "Cleared as false positive"})
                self._send({"ok": True, "alert": alert})
            elif action == "create_case":
                cid = f"CAS-{random.randint(500,999)}"
                alert["status"] = "review"
                alert["case_id"] = cid
                alert["officer"] = officer
                new_case = {
                    "id": cid, "entity": alert["entity"],
                    "alerts": [alert_id], "alert_count": 1,
                    "priority": alert["priority"], "status": "open",
                    "officer": officer,
                    "opened": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sar_due": (datetime.now().replace(day=datetime.now().day)).strftime("%Y-%m-%d"),
                    "typology": alert["typology"],
                    "narrative": notes or f"Case opened from alert {alert_id}.",
                    "sar_status": "under_review",
                }
                CASES.insert(0, new_case)
                _log({"user": officer, "action": "CASE_CREATED",
                      "target": cid, "detail": f"From alert {alert_id}"})
                self._send({"ok": True, "case_id": cid, "case": new_case})
            elif action == "escalate":
                alert["priority"] = "critical"
                _log({"user": officer, "action": "ALERT_ESCALATED",
                      "target": alert_id, "detail": notes})
                self._send({"ok": True, "alert": alert})
            elif action == "add_note":
                alert["notes"] = (alert.get("notes","") + "\n" + notes).strip()
                self._send({"ok": True, "alert": alert})
            else:
                self._send({"error": "Unknown action"}, 400)

        # ── Case update ──────────────────────────────────────────────────────
        elif len(parts) == 4 and parts[1] == "cases" and parts[3] == "update":
            case_id = parts[2]
            case = next((c for c in CASES if c["id"] == case_id), None)
            if not case:
                self._send({"error": "Case not found"}, 404); return
            for field in ["status", "narrative", "officer", "sar_status"]:
                if field in body:
                    case[field] = body[field]
            if body.get("sar_status") == "filed":
                sar = {
                    "id": f"SAR-{random.randint(52,999):04d}",
                    "case_id": case_id,
                    "entity": case["entity"],
                    "amount": 0,
                    "typology": case["typology"],
                    "filed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "officer": body.get("officer", case["officer"]),
                    "status": "filed",
                    "fincen_ref": f"FIN-2024-{random.randint(100,999)}",
                }
                SAR_RECORDS.insert(0, sar)
                _log({"user": body.get("officer","system"), "action": "SAR_FILED",
                      "target": case_id, "detail": f"FinCEN ref: {sar['fincen_ref']}"})
            _log({"user": body.get("officer","system"), "action": "CASE_UPDATED",
                  "target": case_id, "detail": f"Status → {body.get('status','')}"})
            self._send({"ok": True, "case": case})

        # ── Model retrain ────────────────────────────────────────────────────
        elif route == "models/retrain":
            def do_retrain():
                engine = get_engine()
                engine.train()
                _log({"user": "system", "action": "MODEL_RETRAINED",
                      "target": "ensemble", "detail": "Manual retrain triggered"})
            t = threading.Thread(target=do_retrain)
            t.start()
            self._send({"ok": True, "message": "Retraining started in background"})

        else:
            self._send({"error": "Not found"}, 404)


def run(port=None):
    # Railway (and Render/Cloud Run) inject PORT as an environment variable.
    # Fall back to 8787 for local development.
    port = port or int(os.environ.get('PORT', PORT))
    print(f"\n{'='*55}")
    print(f"  AML-TMS Backend API")
    print(f"  Starting on port {port}")
    print(f"{'='*55}")
    print("  Initialising ML engine (training models)...")
    engine = get_engine()  # Train upfront
    print(f"  ML engine ready. Models: iso, xgb, gnn, lstm")
    print(f"  Frontend served at /")
    print(f"  API health: /api/health")
    print(f"{'='*55}\n")
    server = HTTPServer(("0.0.0.0", port), AMLHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', PORT))
    run(port)

# ── Additional endpoints (appended) ──────────────────────────────────────────
# Monkey-patch the GET handler to add new routes

_original_get = AMLHandler.do_GET

def _extended_get(self):
    parsed = urlparse(self.path)
    path = parsed.path.rstrip("/")
    parts = [p for p in path.split("/") if p]

    if not parts or parts[0] != "api":
        _original_get(self)
        return

    route = "/".join(parts[1:])

    if route == "features/summary":
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from feature_engineering import get_feature_summary
        self._send(get_feature_summary())

    elif route == "monitor/report":
        from model_monitor import generate_monitoring_report
        self._send(generate_monitoring_report())

    elif len(parts) == 4 and parts[1] == "sar" and parts[2] == "generate":
        # /api/sar/generate/{case_id}
        case_id = parts[3] if len(parts) > 3 else None
        if case_id:
            case = next((c for c in CASES if c["id"] == case_id), None)
            if case:
                from sar_engine import generate_sar, get_sar_checklist
                case_alerts = [a for a in ALERTS if a.get("case_id") == case_id]
                sar = generate_sar(case, case_alerts, case.get("officer", "AML Officer"))
                checklist = get_sar_checklist(sar)
                self._send({"sar": sar, "checklist": checklist})
                return
        self._send({"error": "Case not found"}, 404)

    else:
        _original_get(self)

AMLHandler.do_GET = _extended_get
