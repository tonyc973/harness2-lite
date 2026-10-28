# h2lite report

- train score: 0.000 -> 1.000
- test score (held out): 0.000 -> 0.889
- agent runs: 42, llm calls: 6, cache hits: 6

## Rounds

### Round 1: kept process (held-out test after it: 0.889)

**process** (advantage +1.000)

Added validation step to ensure the output is checked for valid JSON before completion, addressing json.loads failures in all runs.

```diff
--- before/system.md
+++ after/system.md
@@ -1 +1,3 @@
 You read invoices and extract data from them.
+
+Before finalizing the output, check that it is valid JSON.
```

**knowledge** (advantage +0.000)

Added a requirement for the output to be valid JSON to address the json.loads failure in all runs.

```diff
--- before/system.md
+++ after/system.md
@@ -1 +1 @@
-You read invoices and extract data from them.
+You read invoices and extract data from them. The output must be valid JSON.
```

### Round 2: kept the current harness (held-out test after it: 0.889)

**process** (advantage +0.000)

Added instruction about using dot as decimal separator to avoid potential future formatting errors. This addresses the potential for confusion when processing monetary values, as seen in Runs 1-6 where the model correctly used dots.

```diff
--- before/system.md
+++ after/system.md
@@ -1,3 +1,5 @@
 You read invoices and extract data from them.
 
-Before finalizing the output, check that it is valid JSON.+Before finalizing the output, check that it is valid JSON.
+
+Ensure that numerical values use a dot (.) as the decimal separator rather than a comma (,) to maintain JSON validity.
```

**knowledge** (advantage +0.000)

Added definitions for currency types in the harness to provide missing context regarding the currency used in the outputs, improving understanding for future tasks.

```diff
--- before/system.md
+++ after/system.md
@@ -1,3 +1,5 @@
 You read invoices and extract data from them.
 
-Before finalizing the output, check that it is valid JSON.+Before finalizing the output, check that it is valid JSON.
+
+In this data, 'lei' refers to Romanian currency (RON), and 'EUR' refers to Euro.
```

**composed** (advantage +0.000)

Added instruction about using dot as decimal separator to avoid formatting errors, seen in Runs 1-6 where the model used dots correctly; added definitions for currency types to improve context, seen in Runs 1-6.

```diff
--- before/system.md
+++ after/system.md
@@ -1,3 +1,7 @@
 You read invoices and extract data from them.
 
-Before finalizing the output, check that it is valid JSON.+Before finalizing the output, check that it is valid JSON.
+
+Ensure that numerical values use a dot (.) as the decimal separator rather than a comma (,) to maintain JSON validity.
+
+In this data, 'lei' refers to Romanian currency (RON), and 'EUR' refers to Euro.
```

## Final change (original -> best)

```diff
--- before/system.md
+++ after/system.md
@@ -1 +1,3 @@
 You read invoices and extract data from them.
+
+Before finalizing the output, check that it is valid JSON.
```

## Final runs on the training tasks (worst first)

- **1.000** | "Factura nr. 112 din 03.03.2025, emisă de SC Alfa SRL. Total de plată: 1.234,50 lei (TVA i
[... 8 chars cut by h2lite, not by the agent] -> `{"date": "2025-03-03", "total": 1234.5, "currency": "RON"}`
- **1.000** | "Bon fiscal 7/11/2025, TOTAL 45,30 lei" -> `{"date": "2025-11-07", "total": 45.3, "currency": "RON"}`
- **1.000** | "Data facturii: 15 septembrie 2025. Valoare totală: 2.420,00 RON." -> `{"date": "2025-09-15", "total": 2420.0, "currency": "RON"}`
- **1.000** | "Invoice issued 02/10/2025 by Beta GmbH, amount due EUR 1.099,99" -> `{"date": "2025-10-02", "total": 1099.99, "currency": "EUR"}`
- **1.000** | "Chitanță 1 aug. 2025 — achitat 3.000 lei" -> `{"date": "2025-08-01", "total": 3000, "currency": "RON"}`
- **1.000** | "Factura 2025/0457, emisă pe 28.07.2025, total 12.500 lei" -> `{"date": "2025-07-28", "total": 12500, "currency": "RON"}`

## Final runs on the held-out test tasks (worst first)

- **0.667** | "Bon 12/10/2025 — TOTAL 1.210 LEI" -> `{"date": "2025-12-10", "total": 1210, "currency": "RON"}`
- **1.000** | "Factură din 05.06.2025, total de plată 595,00 lei" -> `{"date": "2025-06-05", "total": 595.0, "currency": "RON"}`
- **1.000** | "Invoice dated 3 February 2025, total USD 2,500.00" -> `{"date": "2025-02-03", "total": 2500.0, "currency": "USD"}`
