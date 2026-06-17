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
output_dir = r"./outputs/ipa_results"
cta_results_dir = r"./outputs/cta_results"
clinical_excel_path = r"./clinical_cases.xlsx"
he_predictions_excel_path = r"./he_predictions.xlsx"
he_predictions_output_dir = os.path.join(output_dir, "he_predictions")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(he_predictions_output_dir, exist_ok=True)


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
#                  JSON extraction utility
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
            probability = float(item.get("probability", 0))
        except:
            probability = 0.0
        cleaned.append({"name": name, "probability": probability})
    return sorted(cleaned, key=lambda x: x.get("probability", 0), reverse=True)[:top_k]


def extract_clinical_text_from_cta_prompt(prompt):
    if not prompt:
        return ""
    m = re.search(
        r"Patient information:\s*(.*?)\n\s*Output strictly in the following JSON format",
        prompt,
        flags=re.S | re.I,
    )
    if m:
        return m.group(1).strip()
    return ""


def load_clinical_info_from_cta(case_id, cta_dir=cta_results_dir):
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
        raise FileNotFoundError(f"CTA result not found for case {case_id}: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed = data.get("rag_parsed_json", {}) or {}
    predictions = top_predictions(parsed.get("predictions", []), top_k=10)
    clinical_text, clinical_info_path = load_clinical_info_from_cta(case_id, cta_dir=cta_dir)
    if not clinical_text:
        clinical_text = extract_clinical_text_from_cta_prompt(data.get("rag_prompt_used", ""))

    return {
        "case_id": str(parsed.get("case_id", case_id)).strip() or case_id,
        "clinical_text": clinical_text,
        "clinical_info_path": clinical_info_path,
        "cta_predictions": predictions,
        "raw": data,
    }


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
        probability = item.get("probability", None)
        if probability is None:
            probability = item.get("prob", None)
        try:
            lines.append(f"- {name}: {float(probability):.4f}")
        except:
            lines.append(f"- {name}")
    return "\n".join(lines)


def parse_he_predictions(text):
    text = text.strip()
    if not text:
        return []

    parsed = extract_json_from_response(text)
    if isinstance(parsed, list):
        return top_predictions(parsed, top_k=20)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("predictions"), list):
            return top_predictions(parsed.get("predictions"), top_k=20)
        if parsed.get("name"):
            return top_predictions([parsed], top_k=20)

    predictions = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-").strip()
        if not line:
            continue
        if ":" in line:
            name, prob = line.rsplit(":", 1)
        elif "," in line:
            name, prob = line.rsplit(",", 1)
        else:
            predictions.append({"name": line, "probability": 0.0})
            continue
        name = name.strip()
        prob = prob.strip()
        try:
            probability = float(prob)
        except:
            probability = 0.0
        predictions.append({"name": name, "probability": probability})

    return top_predictions(predictions, top_k=20)


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


def save_he_predictions(case_id, he_predictions):
    record = {
        "case_id": str(case_id).strip(),
        "he_predictions": top_predictions(he_predictions, top_k=20),
    }
    output_path = os.path.join(he_predictions_output_dir, f"{case_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"[OK] H&E predictions saved: {output_path}")
    return output_path


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


def build_clinical_query(case_id, he_predictions, clinical_info):
    lines = []

    if isinstance(clinical_info, str):
        clinical_text = clinical_info.strip()
    else:
        sex = str(clinical_info.get("sex", "")).strip()
        age = str(clinical_info.get("age", "")).strip()
        chief_complaint = str(clinical_info.get("chief_complaint", "")).strip()
        history = str(clinical_info.get("history_after_removing_pathology_information", "")).strip()
        imaging = str(clinical_info.get("imaging", "")).strip()

        clinical_parts = []
        if sex or age:
            clinical_parts.append(f"Patient sex: {sex}; age: {age}.")
        if chief_complaint:
            clinical_parts.append(f"Chief complaint: {chief_complaint}")
        if history:
            clinical_parts.append(f"History of present illness: {history}")
        if imaging:
            clinical_parts.append(f"Imaging findings: {imaging}")
        clinical_text = "\n".join(clinical_parts)

    if clinical_text:
        lines.append(clinical_text)

    he_text = format_prediction_list(he_predictions)
    if he_text:
        lines.append("H&E image model predictions:")
        lines.append(he_text)

    return "\n".join(lines)


# ============================================================
#                    Initialize DeepSeek Reasoner
# ============================================================
client = OpenAI(
    api_key="YOUR-KEY",
    base_url="https://api.deepseek.com"
)


def build_ipa_prompt(case_id, clinical_query, cta_predictions, he_predictions):
    cta_prediction_text = format_prediction_list(cta_predictions)
    he_prediction_text = format_prediction_list(he_predictions)

    return f"""
    You are an experienced hematopathology diagnostic expert.

    The following information represents AI-assisted multimodal predictions for this case, integrating:
    - Clinical information
    - Imaging information
    - The CTA model output based on chief complaint, clinical history, and imaging
    - The H&E pathology image model output

    These inputs provide several likely and mutually confusable lymphoma subtype candidates.

    --------------------------------------
    Patient clinical information:
    {clinical_query}

    --------------------------------------
    CTA model prediction based on clinical text and imaging:
    {cta_prediction_text}

    Note:
    The CTA result reflects a preliminary judgment based on clinical text and imaging before reviewing the H&E slide.

    --------------------------------------
    H&E pathology image model prediction:
    {he_prediction_text}

    Note:
    The H&E result reflects a morphology-based judgment from histopathologic image features.

    --------------------------------------

    Based on the diagnostic uncertainty among the candidate lymphoma subtypes above, design a complete, tiered immunohistochemistry (IHC) testing strategy that follows standard WHO/ICC diagnostic logic and can guide the pathology department toward a standardized diagnostic workup.

    ### Diagnostic strategy principles:
    1. The purpose of IHC is to resolve the differential diagnostic conflict, not to repeat the AI conclusion.
    2. The recommended IHC strategy should proceed from:
       - Lineage confirmation (B-cell vs T-cell vs NK-cell)
       - Key subtype separation
       - Characteristic supportive or exclusionary evidence
    3. Each IHC marker group must clearly state whether it is used to confirm, distinguish, or exclude candidate entities.
    4. The IHC pathway should be consistent with WHO/ICC lymphoma classification logic.
    5. Recommend IHC only.
       - Do not recommend FISH, PCR, NGS, ISH, or other molecular tests.
    6. The IHC panel should cover the current 2 to 4 most likely confusable subtypes, rather than providing only a generic minimal lymphoma panel.

    --------------------------------------

    Output strictly in the following JSON format. Do not include any extra content, explanations, or natural language outside the JSON:

    {{
      "case_id": "{case_id}",

      "most_confusing_subtypes": [
        "subtype_1",
        "subtype_2",
        "subtype_3"
      ],

      "key_differential_features": "Summarize the key differential diagnostic challenges among these candidate subtypes, including lineage, immunophenotype, and architectural or morphologic features.",

      "recommended_IHCs": [
        {{
          "step": 1,
          "markers": ["CD20", "CD3", "PAX5", "CD79a"],
          "purpose": "Confirm the major lineage of the neoplastic cells, such as B-cell lineage versus T/NK-cell lineage."
        }},
        {{
          "step": 2,
          "markers": ["CD5", "CD10", "BCL6", "MUM1"],
          "purpose": "Distinguish the major candidate subtypes after lineage confirmation and narrow the diagnostic range."
        }},
        {{
          "step": 3,
          "markers": ["Cyclin D1", "CD30", "ALK"],
          "purpose": "Provide supportive or exclusionary evidence for specific high-priority or characteristic entities."
        }},
        {{
          "step": 4,
          "markers": ["Ki-67", "CD21", "PD-1"],
          "purpose": "Assess proliferative activity and tumor microenvironment features to support the final diagnosis."
        }}
      ],

      "final_IHC_panel": [
        "CD20", "CD3", "PAX5", "CD79a",
        "CD5", "CD10", "BCL6", "MUM1",
        "Cyclin D1", "CD30", "ALK",
        "Ki-67", "CD21", "PD-1"
      ]
    }}
    """


def call_deepseek(prompt):
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {
                "role": "system",
                "content": "You are a hematopathology expert. Recommend a targeted immunohistochemistry panel to resolve lymphoma differential diagnoses based on clinical, CTA, and H&E model predictions."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=7000
    )
    return response.choices[0].message.content


def run_ipa_agent(case_id, clinical_info, cta_predictions, he_predictions, save_result=True, output_path=None):
    case_id = str(case_id).strip()
    he_predictions_path = None
    if save_result:
        he_predictions_path = save_he_predictions(case_id, he_predictions)

    clinical_query = build_clinical_query(case_id, he_predictions, clinical_info)
    prompt = build_ipa_prompt(case_id, clinical_query, cta_predictions, he_predictions)

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
                "he_predictions_path": he_predictions_path,
                "rag_prompt_used": prompt,
                "rag_reply_raw": reply_text,
                "rag_parsed_json": None,
            }

    except Exception as e:
        print(f"[X] DeepSeek call failed: {e}")
        return None

    result = {
        "case_id": case_id,
        "he_predictions_path": he_predictions_path,
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


def run_ipa_agent_from_cta(case_id, he_predictions, cta_dir=cta_results_dir, save_result=True, output_path=None):
    cta_result = load_cta_result(case_id, cta_dir=cta_dir)
    return run_ipa_agent(
        case_id=case_id,
        clinical_info=cta_result["clinical_text"],
        cta_predictions=cta_result["cta_predictions"],
        he_predictions=he_predictions,
        save_result=save_result,
        output_path=output_path,
    )


def read_multiline(prompt_text):
    print(prompt_text)
    print("Enter one or more lines. Use format 'diagnosis: probability' when possible. Type END when finished.")
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


def collect_he_predictions_from_dialog():
    text = read_multiline(
        "Enter H&E model prediction(s). Examples: 'Diffuse large B-cell lymphoma, not otherwise specified: 0.82' or a JSON list."
    )
    return parse_he_predictions(text)


def run_interactive(cta_dir=cta_results_dir):
    while True:
        case_id = input("\nPathology number / case ID: ").strip()
        if not case_id:
            print("Case ID is required.")
            continue

        try:
            cta_result = load_cta_result(case_id, cta_dir=cta_dir)
        except Exception as e:
            print(f"[X] {e}")
            again = input("Try another case? Enter y to continue, or any other key to exit: ").strip().lower()
            if again != "y":
                break
            continue

        print("\n--- Clinical information loaded from CTA result ---")
        print(cta_result["clinical_text"] or "(No clinical text found in CTA prompt.)")

        print("\n--- CTA top predictions loaded from CTA result ---")
        print(format_prediction_list(cta_result["cta_predictions"]) or "(No CTA predictions found.)")

        he_predictions = collect_he_predictions_from_dialog()
        if not he_predictions:
            print("No valid H&E prediction was entered. Canceled.")
        else:
            print(f"\n=== Processing case: {case_id} ===")
            result = run_ipa_agent(
                case_id=case_id,
                clinical_info=cta_result["clinical_text"],
                cta_predictions=cta_result["cta_predictions"],
                he_predictions=he_predictions,
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


def run_batch(
    clinical_excel=clinical_excel_path,
    he_excel=he_predictions_excel_path,
    cta_dir=cta_results_dir,
    cta_excel=None,
):
    clinical_df = load_clinical_dataframe(clinical_excel)
    print(clinical_df.head())

    clinical_bank = {}
    for _, r in clinical_df.iterrows():
        case_id = normalize_case_id(r["case_id"])
        if case_id:
            clinical_bank[case_id] = r

    he_bank = load_he_predictions_from_excel(he_excel)
    cta_excel_bank = load_cta_predictions_from_excel(cta_excel)

    for _, row in clinical_df.iterrows():
        case_id = normalize_case_id(row["case_id"])
        if not case_id:
            continue

        print(f"\n=== Processing case: {case_id} ===")

        output_path = os.path.join(output_dir, f"{case_id}.json")
        if os.path.exists(output_path):
            print("Already generated. Skipping.")
            continue

        he_predictions = he_bank.get(case_id, [])

        try:
            cta_result = load_cta_result(case_id, cta_dir=cta_dir)
            clinical_info = cta_result["clinical_text"] or clinical_bank.get(case_id, {})
            cta_predictions = cta_result["cta_predictions"]
        except Exception:
            clinical_info = clinical_bank.get(case_id, {})
            cta_predictions = cta_excel_bank.get(case_id, [])

        run_ipa_agent(
            case_id=case_id,
            clinical_info=clinical_info,
            cta_predictions=cta_predictions,
            he_predictions=he_predictions,
            save_result=True,
            output_path=output_path,
        )
        time.sleep(1)

    print("\n=== All processing completed ===")


def parse_args():
    parser = argparse.ArgumentParser(
        description="IPA agent: recommend an immunohistochemistry panel based on CTA and H&E model predictions."
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive input")
    parser.add_argument("--clinical-excel", default=clinical_excel_path, help="Clinical Excel path for batch mode")
    parser.add_argument("--he-excel", default=he_predictions_excel_path, help="H&E prediction Excel path for batch mode")
    parser.add_argument("--cta-dir", default=cta_results_dir, help="Directory containing CTA JSON outputs")
    parser.add_argument("--cta-excel", default=None, help="Optional CTA prediction Excel path for batch fallback")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.interactive:
        run_interactive(cta_dir=args.cta_dir)
    else:
        run_batch(
            clinical_excel=args.clinical_excel,
            he_excel=args.he_excel,
            cta_dir=args.cta_dir,
            cta_excel=args.cta_excel,
        )
