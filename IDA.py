import os
import json
import time
import re
import argparse
import pandas as pd
import numpy as np
from openai import OpenAI

# ============================================================
#                 Configuration
# ============================================================
output_dir = r"./outputs/ida_results"
cta_results_dir = r"./outputs/cta_results"
ipa_results_dir = r"./outputs/ipa_results"
mwpa_results_dir = r"./outputs/mwpa_results"
clinical_excel_path = r"./clinical_cases.xlsx"
he_predictions_excel_path = r"./he_predictions.xlsx"
molecular_results_output_dir = os.path.join(output_dir, "molecular_results")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(molecular_results_output_dir, exist_ok=True)


LYMPHOMA_LABELS = [
    "Anaplastic large cell lymphoma, ALK-negative",
    "Anaplastic large cell lymphoma, ALK-positive",
    "Angioimmunoblastic T-cell lymphoma",
    "B-lymphoblastic leukemia/lymphoma",
    "Burkitt lymphoma",
    "Chronic lymphocytic leukemia/small lymphocytic lymphoma",
    "Classic Hodgkin lymphoma, lymphocyte-rich subtype",
    "Classic Hodgkin lymphoma, mixed cellularity subtype",
    "Classic Hodgkin lymphoma, nodular sclerosis subtype",
    "Diffuse large B-cell lymphoma, not otherwise specified",
    "Extranodal NK/T-cell lymphoma",
    "Follicular lymphoma",
    "High-grade B-cell lymphoma with MYC and BCL2 rearrangements",
    "Nodal marginal zone lymphoma",
    "Extranodal marginal zone lymphoma of mucosa-associated lymphoid tissue (MALT lymphoma)",
    "Mantle cell lymphoma",
    "Nodular lymphocyte-predominant Hodgkin lymphoma",
    "Peripheral T-cell lymphoma, NOS",
    "Plasmacytoma",
    "T-lymphoblastic lymphoma",
]


# ============================================================
#                  Shared utilities
# ============================================================
def extract_json_from_response(text):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass
    text2 = re.sub(r'^```(?:json)?', '', text, flags=re.I)
    text2 = re.sub(r'```$', '', text2, flags=re.M).strip()
    try:
        return json.loads(text2)
    except:
        pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            return None
    return None


def normalize_case_id(value):
    if pd.isna(value):
        return None
    try:
        return str(int(float(value))).strip()
    except:
        return str(value).strip()


def top_predictions(predictions, top_k=10):
    if not isinstance(predictions, list):
        return []
    cleaned = []
    for item in predictions:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            probability = float(item.get("probability", item.get("prob", 0)))
        except:
            probability = 0.0
        cleaned.append({"name": name, "probability": probability})
    return sorted(cleaned, key=lambda x: x.get("probability", 0), reverse=True)[:top_k]


def format_prediction_list(predictions):
    if isinstance(predictions, str):
        return predictions.strip()
    if not isinstance(predictions, list):
        return ""

    lines = []
    for item in predictions:
        if isinstance(item, str):
            if item.strip():
                lines.append(f"- {item.strip()}")
            continue
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        probability = item.get("probability", item.get("prob", None))
        try:
            lines.append(f"- {name}: {float(probability):.4f}")
        except:
            lines.append(f"- {name}")
    return "\n".join(lines)


def format_marker_results(results, empty_message):
    if not results:
        return empty_message
    if isinstance(results, str):
        return results.strip() or empty_message
    if isinstance(results, dict):
        items = []
        for marker, result in results.items():
            marker = str(marker).strip()
            result = str(result).strip()
            if marker:
                items.append(f"{marker}: {result}")
        return "; ".join(items) if items else empty_message
    if isinstance(results, list):
        items = []
        for item in results:
            if isinstance(item, str):
                if item.strip():
                    items.append(item.strip())
            elif isinstance(item, dict):
                test = str(item.get("test", item.get("marker", ""))).strip()
                result = str(item.get("result", "")).strip()
                if test:
                    items.append(f"{test}: {result}")
        return "; ".join(items) if items else empty_message
    return str(results).strip()


def parse_test_results(text):
    text = text.strip()
    if not text:
        return []

    parsed = extract_json_from_response(text)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("molecular_results"), list):
            return parsed.get("molecular_results")
        if isinstance(parsed.get("results"), list):
            return parsed.get("results")
        return [
            {"test": str(test).strip(), "result": str(result).strip()}
            for test, result in parsed.items()
            if str(test).strip()
        ]
    if isinstance(parsed, list):
        return parsed

    results = []
    chunks = re.split(r"[;\n]+", text)
    for raw_line in chunks:
        line = raw_line.strip().lstrip("-").strip()
        if not line:
            continue
        if ":" in line:
            test, result = line.split(":", 1)
        else:
            test, result = line, ""
        test = test.strip()
        result = result.strip()
        if test:
            results.append({"test": test, "result": result})
    return results


def normalize_test_panel(panel):
    if not panel:
        return []
    if isinstance(panel, str):
        tests = re.split(r"[,;\n]+", panel)
    elif isinstance(panel, list):
        tests = panel
    else:
        return []

    normalized = []
    seen = set()
    for item in tests:
        if isinstance(item, dict):
            item = item.get("test", item.get("name", item.get("marker", "")))
        item = str(item).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized


def he_probabilities_to_predictions(probabilities, top_k=10):
    probs = np.array(probabilities, dtype=float)
    if probs.size == 0:
        return []
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [
        {"name": LYMPHOMA_LABELS[idx], "probability": float(probs[idx])}
        for idx in top_idx
        if 0 <= idx < len(LYMPHOMA_LABELS)
    ]


# ============================================================
#                  Load upstream outputs
# ============================================================
def extract_clinical_text_from_cta_prompt(prompt):
    if not prompt:
        return ""
    m = re.search(
        r"Patient Information:\s*(.*?)\n\s*-{5,}\s*\n\s*Return ONLY valid JSON",
        prompt,
        flags=re.S | re.I,
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r"Patient information:\s*(.*?)\n\s*Output strictly in the following JSON format",
        prompt,
        flags=re.S | re.I,
    )
    if m:
        return m.group(1).strip()
    return ""


def load_clinical_info(case_id, cta_dir=cta_results_dir):
    clinical_path = os.path.join(cta_dir, "clinical_info", f"{case_id}.json")
    if not os.path.exists(clinical_path):
        return "", None

    with open(clinical_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parts = []
    sex = str(data.get("sex", "")).strip()
    age = str(data.get("age", "")).strip()
    chief_complaint = str(data.get("chief_complaint", "")).strip()
    history = str(data.get("history_after_removing_pathology_information", "")).strip()
    imaging = str(data.get("imaging", "")).strip()

    if sex or age:
        parts.append(f"Patient sex: {sex}; age: {age}.")
    if chief_complaint:
        parts.append(f"Chief complaint: {chief_complaint}")
    if history:
        parts.append(f"History of present illness: {history}")
    if imaging:
        parts.append(f"Imaging findings: {imaging}")

    return "\n".join(parts), clinical_path


def load_cta_result(case_id, cta_dir=cta_results_dir):
    path = os.path.join(cta_dir, f"{case_id}.json")
    if not os.path.exists(path):
        return {"cta_predictions": [], "clinical_text": "", "path": None, "clinical_info_path": None}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed = data.get("rag_parsed_json", {}) or {}
    predictions = top_predictions(parsed.get("predictions", []), top_k=10)
    clinical_text, clinical_info_path = load_clinical_info(case_id, cta_dir=cta_dir)
    if not clinical_text:
        clinical_text = extract_clinical_text_from_cta_prompt(data.get("rag_prompt_used", ""))

    return {
        "cta_predictions": predictions,
        "clinical_text": clinical_text,
        "path": path,
        "clinical_info_path": clinical_info_path,
        "raw": data,
    }


def load_he_predictions(case_id, ipa_dir=ipa_results_dir):
    path = os.path.join(ipa_dir, "he_predictions", f"{case_id}.json")
    if not os.path.exists(path):
        return {"he_predictions": [], "path": None}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "he_predictions": top_predictions(data.get("he_predictions", []), top_k=10),
        "path": path,
        "raw": data,
    }


def load_ipa_result(case_id, ipa_dir=ipa_results_dir):
    path = os.path.join(ipa_dir, f"{case_id}.json")
    if not os.path.exists(path):
        return {"ipa_result": {}, "path": None}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed = data.get("rag_parsed_json", {}) or {}
    return {"ipa_result": parsed, "path": path, "raw": data}


def load_ihc_results(case_id, mwpa_dir=mwpa_results_dir):
    path = os.path.join(mwpa_dir, "ihc_results", f"{case_id}.json")
    if not os.path.exists(path):
        return {"ihc_results": [], "recommended_IHC_panel": [], "path": None}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "ihc_results": data.get("ihc_results", []),
        "recommended_IHC_panel": data.get("recommended_IHC_panel", []),
        "path": path,
        "raw": data,
    }


def load_mwpa_result(case_id, mwpa_dir=mwpa_results_dir):
    path = os.path.join(mwpa_dir, f"{case_id}.json")
    if not os.path.exists(path):
        return {"mwpa_result": {}, "mwpa_predictions": [], "suggested_next_tests": [], "path": None}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed = data.get("rag_parsed_json", {}) or {}
    return {
        "mwpa_result": parsed,
        "mwpa_predictions": top_predictions(parsed.get("final_prediction", []), top_k=10),
        "suggested_next_tests": parsed.get("suggested_next_tests", []),
        "path": path,
        "raw": data,
    }


def load_upstream_context(case_id, cta_dir=cta_results_dir, ipa_dir=ipa_results_dir, mwpa_dir=mwpa_results_dir):
    cta = load_cta_result(case_id, cta_dir=cta_dir)
    he = load_he_predictions(case_id, ipa_dir=ipa_dir)
    ipa = load_ipa_result(case_id, ipa_dir=ipa_dir)
    ihc = load_ihc_results(case_id, mwpa_dir=mwpa_dir)
    mwpa = load_mwpa_result(case_id, mwpa_dir=mwpa_dir)

    return {
        "case_id": case_id,
        "clinical_text": cta.get("clinical_text", ""),
        "cta_predictions": cta.get("cta_predictions", []),
        "he_predictions": he.get("he_predictions", []),
        "ipa_result": ipa.get("ipa_result", {}),
        "ihc_results": ihc.get("ihc_results", []),
        "recommended_IHC_panel": ihc.get("recommended_IHC_panel", []),
        "mwpa_result": mwpa.get("mwpa_result", {}),
        "mwpa_predictions": mwpa.get("mwpa_predictions", []),
        "suggested_next_tests": mwpa.get("suggested_next_tests", []),
        "source_paths": {
            "cta_result": cta.get("path"),
            "clinical_info": cta.get("clinical_info_path"),
            "he_predictions": he.get("path"),
            "ipa_result": ipa.get("path"),
            "ihc_results": ihc.get("path"),
            "mwpa_result": mwpa.get("path"),
        },
    }


def save_molecular_results(case_id, molecular_results, recommended_tests=None):
    record = {
        "case_id": str(case_id).strip(),
        "recommended_molecular_tests": normalize_test_panel(recommended_tests),
        "molecular_results": molecular_results,
        "molecular_status": "not_performed" if not molecular_results else "provided",
    }
    output_path = os.path.join(molecular_results_output_dir, f"{case_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"[OK] Molecular results saved: {output_path}")
    return output_path


# ============================================================
#                    Initialize DeepSeek Reasoner
# ============================================================
client = OpenAI(
    api_key="YOUR-KEY",
    base_url="https://api.deepseek.com"
)


def build_ida_prompt(case_id, upstream_context, molecular_results):
    clinical_text = upstream_context.get("clinical_text", "")
    cta_text = format_prediction_list(upstream_context.get("cta_predictions", []))
    he_text = format_prediction_list(upstream_context.get("he_predictions", []))
    mwpa_text = format_prediction_list(upstream_context.get("mwpa_predictions", []))
    ihc_text = format_marker_results(
        upstream_context.get("ihc_results", []),
        "IHC was not performed or no IHC result was provided."
    )
    molecular_text = format_marker_results(
        molecular_results,
        "Molecular, cytogenetic, and in situ hybridization testing was not performed or no result was provided."
    )
    mwpa_result = upstream_context.get("mwpa_result", {}) or {}
    mwpa_summary = json.dumps(
        {
            "key_evidence": mwpa_result.get("key_evidence", ""),
            "diagnostic_conclusion": mwpa_result.get("diagnostic_conclusion", ""),
            "diagnostic_certainty": mwpa_result.get("diagnostic_certainty", ""),
            "overall_confidence": mwpa_result.get("overall_confidence", 0),
            "suggested_next_tests": mwpa_result.get("suggested_next_tests", []),
        },
        ensure_ascii=False,
        indent=2,
    )

    return f"""
    You are an experienced hematopathologist who routinely performs final lymphoma sign-out and multidisciplinary diagnostic review.

    The following information represents the currently available multimodal evidence for this case. Please synthesize the data according to final diagnostic standards in hematopathology.

    --------------------------------------
    [1] Patient clinical information:
    {clinical_text}

    --------------------------------------
    [2] CTA model predictions based on clinical text and imaging:
    {cta_text}

    --------------------------------------
    [3] H&E pathology image model predictions:
    {he_text}

    --------------------------------------
    [4] MWPA integrated prediction after CTA, H&E, IPA, and IHC evidence:
    {mwpa_text}

    MWPA diagnostic summary:
    {mwpa_summary}

    --------------------------------------
    [5] Completed immunohistochemistry results:
    {ihc_text}

    --------------------------------------
    [6] Completed molecular, cytogenetic, pathogen-associated, and in situ hybridization results:
    {molecular_text}

    --------------------------------------
    Final diagnostic task:

    Treat the above information as the currently available evidence in a real clinical final sign-out setting. Integrate the case from the following dimensions:
    - Morphologic concordance between H&E prediction and typical histopathologic features.
    - Immunophenotypic compatibility, including lineage assignment and subtype-defining markers.
    - Molecular, cytogenetic, pathogen-associated, or in situ hybridization support or exclusion.
    - Diagnostic exclusivity against major competing lymphoma entities.

    Requirements:
    1. The final diagnosis must be restricted to lymphoma entities appearing in the H&E prediction and MWPA prediction candidate ranges.
    2. Do not introduce a new disease entity that does not appear in those candidate lists, unless all candidates are explicitly unsupported.
    3. Provide the Top 1 to 10 possible diagnoses in descending likelihood.
    4. Assign each diagnosis a probability from 0 to 1 to reflect relative evidentiary support. The probabilities do not need to sum to 1.
    5. Summarize the key final sign-out evidence using pathologic diagnostic language, including morphology, IHC, molecular/cytogenetic findings, and ISH or pathogen-associated results when available.
    6. Provide an integrated final diagnostic impression equivalent to the diagnostic comment in a pathology report.
    7. Assign diagnostic certainty:
       - "high": evidence is highly concordant and sufficient for a final diagnosis.
       - "low": evidence is insufficient for a reliable final diagnosis.
    8. Recommend additional tests only if diagnostic certainty remains insufficient.
       - Do not repeat tests that have already been completed.
       - Recommended tests must have clear differential diagnostic value.

    --------------------------------------
    Output strictly in the following JSON format. Do not include Markdown, explanations, or any text outside the JSON:

    {{
      "case_id": "{case_id}",
      "final_prediction": [
        {{"name": "Peripheral T-cell lymphoma, NOS", "probability": 0.65}},
        {{"name": "Angioimmunoblastic T-cell lymphoma", "probability": 0.25}},
        {{"name": "Anaplastic large cell lymphoma, ALK-negative", "probability": 0.10}}
      ],
      "key_evidence": "Summarize key morphologic, immunophenotypic, molecular, cytogenetic, and ISH evidence supporting the leading final diagnosis.",
      "diagnostic_conclusion": "Provide the integrated final diagnostic impression.",
      "diagnostic_certainty": "low",
      "overall_confidence": 0.00,
      "suggested_next_tests": [
        "TCR gene rearrangement analysis",
        "EBER in situ hybridization"
      ]
    }}
    """


def call_deepseek(prompt):
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {
                "role": "system",
                "content": "You are a hematopathology expert. Integrate clinical, CTA, H&E, IHC, MWPA, molecular, cytogenetic, and ISH evidence to produce a final lymphoma diagnostic impression."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=7000
    )
    return response.choices[0].message.content


def run_ida_agent(case_id, upstream_context, molecular_results=None, save_result=True, output_path=None):
    case_id = str(case_id).strip()
    molecular_results = molecular_results or []
    molecular_results_path = None
    if save_result:
        molecular_results_path = save_molecular_results(
            case_id,
            molecular_results,
            recommended_tests=upstream_context.get("suggested_next_tests", []),
        )

    prompt = build_ida_prompt(case_id, upstream_context, molecular_results)

    try:
        reply_text = call_deepseek(prompt)
        parsed = extract_json_from_response(reply_text)

        if parsed is None:
            if save_result:
                if output_path is None:
                    output_path = os.path.join(output_dir, f"{case_id}.json")
                raw_path = output_path.replace(".json", "_raw.txt")
                with open(raw_path, "w", encoding="utf-8") as rf:
                    rf.write(reply_text)
                print(f"[!] JSON parsing failed. Raw response saved to: {raw_path}")
            return {
                "case_id": case_id,
                "molecular_results_path": molecular_results_path,
                "source_paths": upstream_context.get("source_paths", {}),
                "rag_prompt_used": prompt,
                "rag_reply_raw": reply_text,
                "rag_parsed_json": None,
            }

    except Exception as e:
        print(f"[X] DeepSeek call failed: {e}")
        return None

    result = {
        "case_id": case_id,
        "molecular_results_path": molecular_results_path,
        "source_paths": upstream_context.get("source_paths", {}),
        "rag_prompt_used": prompt,
        "rag_reply_raw": reply_text,
        "rag_parsed_json": parsed,
    }

    if save_result:
        if output_path is None:
            output_path = os.path.join(output_dir, f"{case_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved: {output_path}")

    return result


def run_ida_agent_from_files(case_id, molecular_results=None, cta_dir=cta_results_dir, ipa_dir=ipa_results_dir, mwpa_dir=mwpa_results_dir, save_result=True, output_path=None):
    upstream_context = load_upstream_context(case_id, cta_dir=cta_dir, ipa_dir=ipa_dir, mwpa_dir=mwpa_dir)
    return run_ida_agent(
        case_id=case_id,
        upstream_context=upstream_context,
        molecular_results=molecular_results,
        save_result=save_result,
        output_path=output_path,
    )


# ============================================================
#                  Interactive and batch modes
# ============================================================
def read_multiline(prompt_text):
    print(prompt_text)
    print("Enter one or more lines. Use format 'test: result' when possible. Press Enter on an empty line if testing was not performed. Type END when finished.")
    lines = []
    while True:
        value = input("> ").strip()
        if value.upper() == "END":
            break
        if not value and not lines:
            break
        if not value and lines:
            break
        lines.append(value)
    return "\n".join(lines).strip()


def collect_molecular_results_from_dialog(recommended_tests):
    tests = normalize_test_panel(recommended_tests)

    if not tests:
        text = read_multiline(
            "No MWPA suggested_next_tests were found. Enter completed molecular, cytogenetic, pathogen-associated, or ISH results manually, or leave empty if not performed."
        )
        return parse_test_results(text)

    print("\nEnter the completed result for each MWPA-recommended molecular or ancillary test.")
    print("Press Enter to leave a test blank. If all tests are blank, testing will be treated as not performed.\n")

    results = []
    for test in tests:
        value = input(f"{test} result: ").strip()
        if value:
            results.append({"test": test, "result": value})

    if not results:
        return []
    return results


def run_interactive(cta_dir=cta_results_dir, ipa_dir=ipa_results_dir, mwpa_dir=mwpa_results_dir):
    while True:
        case_id = input("\nPathology number / case ID: ").strip()
        if not case_id:
            print("Case ID is required.")
            continue

        upstream_context = load_upstream_context(case_id, cta_dir=cta_dir, ipa_dir=ipa_dir, mwpa_dir=mwpa_dir)

        print("\n--- Clinical information loaded from upstream outputs ---")
        print(upstream_context.get("clinical_text") or "(No clinical information found.)")

        print("\n--- CTA top predictions ---")
        print(format_prediction_list(upstream_context.get("cta_predictions", [])) or "(No CTA predictions found.)")

        print("\n--- H&E top predictions ---")
        print(format_prediction_list(upstream_context.get("he_predictions", [])) or "(No H&E predictions found.)")

        print("\n--- MWPA top predictions ---")
        print(format_prediction_list(upstream_context.get("mwpa_predictions", [])) or "(No MWPA predictions found.)")

        print("\n--- Completed IHC results loaded from MWPA step ---")
        print(format_marker_results(upstream_context.get("ihc_results", []), "(No IHC results found.)"))

        print("\n--- MWPA suggested molecular or ancillary tests ---")
        suggested_tests = normalize_test_panel(upstream_context.get("suggested_next_tests", []))
        if suggested_tests:
            print("\n".join([f"- {test}" for test in suggested_tests]))
        else:
            print("(No suggested_next_tests found in MWPA result.)")

        molecular_results = collect_molecular_results_from_dialog(suggested_tests)

        print(f"\n=== Processing case: {case_id} ===")
        result = run_ida_agent(
            case_id=case_id,
            upstream_context=upstream_context,
            molecular_results=molecular_results,
        )
        if result and result.get("rag_parsed_json") is not None:
            print("\n--- DeepSeek JSON Result ---")
            print(json.dumps(result["rag_parsed_json"], ensure_ascii=False, indent=2))

        again = input("\nEnter another case? Enter y to continue, or any other key to exit: ").strip().lower()
        if again != "y":
            break

    print("\n=== Interactive input finished ===")


def load_clinical_dataframe(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = [str(c).strip() for c in df.columns]

    positional_columns = [
        "case_id",
        "category",
        "chief_complaint",
        "history_after_removing_pathology_information",
        "imaging",
        "sex",
        "age",
        "pathology",
    ]
    rename_map = {
        "Patient_ID": "case_id",
        "Case_ID": "case_id",
        "case_id": "case_id",
        "pathology_id": "case_id",
        "chief_complaint": "chief_complaint",
        "history_after_removing_pathology_information": "history_after_removing_pathology_information",
        "imaging": "imaging",
        "sex": "sex",
        "age": "age",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    if "case_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "case_id"})

    for idx, target_name in enumerate(positional_columns):
        if target_name in df.columns or idx >= len(df.columns):
            continue
        df = df.rename(columns={df.columns[idx]: target_name})

    return df


def build_clinical_text_from_row(row):
    parts = []
    sex = str(row.get("sex", "")).strip()
    age = str(row.get("age", "")).strip()
    chief_complaint = str(row.get("chief_complaint", "")).strip()
    history = str(row.get("history_after_removing_pathology_information", "")).strip()
    imaging = str(row.get("imaging", "")).strip()

    if sex or age:
        parts.append(f"Patient sex: {sex}; age: {age}.")
    if chief_complaint:
        parts.append(f"Chief complaint: {chief_complaint}")
    if history:
        parts.append(f"History of present illness: {history}")
    if imaging:
        parts.append(f"Imaging findings: {imaging}")
    return "\n".join(parts)


def load_he_predictions_from_excel(excel_path=he_predictions_excel_path):
    if not excel_path or not os.path.exists(excel_path):
        return {}

    df = pd.read_excel(excel_path)
    bank = {}

    for _, row in df.iterrows():
        case_id = normalize_case_id(row.get("Patient_ID", row.iloc[0]))
        if not case_id:
            continue

        if "Probabilities" in row:
            probs = row["Probabilities"]
            if isinstance(probs, str):
                probs = np.array(json.loads(probs))
            else:
                probs = np.array(probs)
            bank[case_id] = he_probabilities_to_predictions(probs, top_k=10)
            continue

        predictions = []
        for col in df.columns[1:]:
            value = row[col]
            if pd.isna(value):
                value = 0.0
            predictions.append({"name": str(col).strip(), "probability": float(value)})
        bank[case_id] = top_predictions(predictions, top_k=10)

    return bank


def load_cta_predictions_from_excel(excel_path):
    if not excel_path or not os.path.exists(excel_path):
        return {}

    df = pd.read_excel(excel_path)
    case_id_col = df.columns[0]
    class_cols = df.columns[1:]
    bank = {}

    for _, row in df.iterrows():
        case_id = normalize_case_id(row[case_id_col])
        if not case_id:
            continue

        preds = []
        for cls in class_cols:
            prob = row[cls]
            if pd.isna(prob):
                prob = 0.0
            preds.append({"name": str(cls).strip(), "probability": float(prob)})
        bank[case_id] = top_predictions(preds, top_k=10)

    return bank


def load_mwpa_predictions_from_excel(excel_path):
    if not excel_path or not os.path.exists(excel_path):
        return {}

    bank = {}
    df = pd.read_excel(excel_path)
    case_id_col = df.columns[0]
    class_cols = df.columns[1:]

    for _, row in df.iterrows():
        case_id = normalize_case_id(row[case_id_col])
        if not case_id:
            continue
        preds = []
        for cls in class_cols:
            prob = row[cls]
            if pd.isna(prob):
                prob = 0.0
            preds.append({"name": str(cls).strip(), "probability": float(prob)})
        bank[case_id] = top_predictions(preds, top_k=10)
    return bank


def load_molecular_results_from_excel(excel_path):
    if not excel_path or not os.path.exists(excel_path):
        return {}

    df = pd.read_excel(excel_path)
    bank = {}

    for _, row in df.iterrows():
        case_id = normalize_case_id(row.iloc[0])
        if not case_id:
            continue

        if "molecular_results" in df.columns:
            value = row["molecular_results"]
        elif len(row) > 1:
            value = row.iloc[1]
        else:
            value = ""

        if pd.isna(value):
            value = ""
        bank[case_id] = parse_test_results(str(value).strip())

    return bank


def run_batch(
    clinical_excel=clinical_excel_path,
    he_excel=he_predictions_excel_path,
    cta_dir=cta_results_dir,
    ipa_dir=ipa_results_dir,
    mwpa_dir=mwpa_results_dir,
    molecular_excel=None,
    cta_excel=None,
    mwpa_excel=None,
):
    clinical_df = load_clinical_dataframe(clinical_excel)
    print(clinical_df.head())

    clinical_bank = {}
    for _, row in clinical_df.iterrows():
        case_id = normalize_case_id(row["case_id"])
        if case_id:
            clinical_bank[case_id] = row

    he_excel_bank = load_he_predictions_from_excel(he_excel)
    cta_excel_bank = load_cta_predictions_from_excel(cta_excel)
    mwpa_excel_bank = load_mwpa_predictions_from_excel(mwpa_excel)
    molecular_bank = load_molecular_results_from_excel(molecular_excel)

    for _, row in clinical_df.iterrows():
        case_id = normalize_case_id(row["case_id"])
        if not case_id:
            continue

        print(f"\n=== Processing case: {case_id} ===")

        output_path = os.path.join(output_dir, f"{case_id}.json")
        if os.path.exists(output_path):
            print("Already generated. Skipping.")
            continue

        upstream_context = load_upstream_context(
            case_id,
            cta_dir=cta_dir,
            ipa_dir=ipa_dir,
            mwpa_dir=mwpa_dir,
        )

        if not upstream_context.get("clinical_text"):
            upstream_context["clinical_text"] = build_clinical_text_from_row(clinical_bank.get(case_id, {}))
        if not upstream_context.get("he_predictions"):
            upstream_context["he_predictions"] = he_excel_bank.get(case_id, [])
        if not upstream_context.get("cta_predictions"):
            upstream_context["cta_predictions"] = cta_excel_bank.get(case_id, [])
        if not upstream_context.get("mwpa_predictions"):
            upstream_context["mwpa_predictions"] = mwpa_excel_bank.get(case_id, [])

        molecular_results = molecular_bank.get(case_id, [])

        run_ida_agent(
            case_id=case_id,
            upstream_context=upstream_context,
            molecular_results=molecular_results,
            save_result=True,
            output_path=output_path,
        )
        time.sleep(1)

    print("\n=== All processing completed ===")


def parse_args():
    parser = argparse.ArgumentParser(
        description="IDA agent: integrate CTA, H&E, IHC, MWPA, and molecular evidence for final lymphoma diagnosis."
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive input")
    parser.add_argument("--clinical-excel", default=clinical_excel_path, help="Clinical Excel path for batch mode")
    parser.add_argument("--he-excel", default=he_predictions_excel_path, help="H&E prediction Excel path for batch fallback")
    parser.add_argument("--cta-dir", default=cta_results_dir, help="Directory containing CTA JSON outputs")
    parser.add_argument("--ipa-dir", default=ipa_results_dir, help="Directory containing IPA JSON outputs")
    parser.add_argument("--mwpa-dir", default=mwpa_results_dir, help="Directory containing MWPA JSON outputs")
    parser.add_argument("--molecular-excel", default=None, help="Optional molecular result Excel path for batch mode")
    parser.add_argument("--cta-excel", default=None, help="Optional CTA prediction Excel path for batch fallback")
    parser.add_argument("--mwpa-excel", default=None, help="Optional MWPA prediction Excel path for batch fallback")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.interactive:
        run_interactive(cta_dir=args.cta_dir, ipa_dir=args.ipa_dir, mwpa_dir=args.mwpa_dir)
    else:
        run_batch(
            clinical_excel=args.clinical_excel,
            he_excel=args.he_excel,
            cta_dir=args.cta_dir,
            ipa_dir=args.ipa_dir,
            mwpa_dir=args.mwpa_dir,
            molecular_excel=args.molecular_excel,
            cta_excel=args.cta_excel,
            mwpa_excel=args.mwpa_excel,
        )
