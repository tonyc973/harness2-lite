"""Real example: an LLM agent that extracts date, total and currency from Romanian invoices.

Needs OPENAI_API_KEY (or any OpenAI-compatible endpoint, see README). Costs cents with gpt-4o-mini.

    h2lite examples/invoice_extract/agent.py --harness examples/invoice_extract/harness \
        --out runs/invoice --rounds 2

Typical traps for a small model: "1.234,50" means 1234.50, "7/11/2025" is day-first in Romania,
"lei" is RON, and JSON wrapped in ``` fences fails a strict parser. The editor sees the runs and
their scores, never the expected answers (they live in EXPECTED, used only by score()).
"""
import json
import os

import h2lite

chat = h2lite.openai_chat(os.environ.get("AGENT_MODEL", "gpt-4o-mini"))

REQUEST = ("Extract the invoice date, the total amount and the currency from the text. Reply with a JSON "
           "object with the keys date (YYYY-MM-DD), total (a number) and currency (ISO 4217 code).\n\nText: ")

EXPECTED = {
    "Factura nr. 112 din 03.03.2025, emisă de SC Alfa SRL. Total de plată: 1.234,50 lei (TVA inclus).":
        {"date": "2025-03-03", "total": 1234.50, "currency": "RON"},
    "Bon fiscal 7/11/2025, TOTAL 45,30 lei": {"date": "2025-11-07", "total": 45.30, "currency": "RON"},
    "Data facturii: 15 septembrie 2025. Valoare totală: 2.420,00 RON.":
        {"date": "2025-09-15", "total": 2420.00, "currency": "RON"},
    "Invoice issued 02/10/2025 by Beta GmbH, amount due EUR 1.099,99":
        {"date": "2025-10-02", "total": 1099.99, "currency": "EUR"},
    "Chitanță 1 aug. 2025 — achitat 3.000 lei": {"date": "2025-08-01", "total": 3000.00, "currency": "RON"},
    "Factura 2025/0457, emisă pe 28.07.2025, total 12.500 lei":
        {"date": "2025-07-28", "total": 12500.00, "currency": "RON"},
    # held out (TEST_TASKS): the last one uses a comma as THOUSANDS separator, to catch an overfitted rule
    "Factură din 05.06.2025, total de plată 595,00 lei": {"date": "2025-06-05", "total": 595.00, "currency": "RON"},
    "Bon 12/10/2025 — TOTAL 1.210 LEI": {"date": "2025-10-12", "total": 1210.00, "currency": "RON"},
    "Invoice dated 3 February 2025, total USD 2,500.00": {"date": "2025-02-03", "total": 2500.00, "currency": "USD"},
}
TASKS = list(EXPECTED)[:6]
TEST_TASKS = list(EXPECTED)[6:]


def run(harness, task):
    raw = chat(REQUEST + task, system=h2lite.as_prompt(harness))
    try:
        data = json.loads(raw)  # strict on purpose: a downstream program would be strict too
    except ValueError as e:
        return raw, f"model replied:\n{raw}\n\njson.loads failed: {e}"
    return json.dumps(data, ensure_ascii=False), f"model replied:\n{raw}\n\nparsed OK"


def score(task, output):
    try:
        got = json.loads(output)
    except ValueError:
        return 0.0
    if not isinstance(got, dict):
        return 0.0
    exp = EXPECTED[task]
    total = got.get("total")
    checks = [
        str(got.get("date")) == exp["date"],
        isinstance(total, (int, float)) and not isinstance(total, bool) and abs(total - exp["total"]) < 0.005,
        str(got.get("currency", "")).upper() == exp["currency"],
    ]
    return sum(checks) / len(checks)
