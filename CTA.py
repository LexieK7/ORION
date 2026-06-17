import os
import json
import time
import re
import argparse
import pandas as pd
from openai import OpenAI

# ============================================================
#                 Configuration
# ============================================================
# Output directory
output_dir = r"outputs/cta_results/"
clinical_info_dir = os.path.join(output_dir, "clinical_info")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(clinical_info_dir, exist_ok=True)

# Clinical Excel file for batch mode
path = r"./clinical_cases.xlsx"


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


# ============================================================
#        Build clinical information for the prompt
# ============================================================
def build_clinical_query(case_id, llm_he_part2, r, extra_ihc, extra_ish, extra_molecular):
    he_top1 = ""
    try:
        m = re.search(r'"([^"]+)"', llm_he_part2)
        if m:
            he_top1 = m.group(1).strip()
    except:
        he_top1 = "No valid diagnosis was extracted from the H&E model."

    chief_complaint = str(r.get("chief_complaint", "")).strip()
    history_filtered = str(r.get("history_after_removing_pathology_information", "")).strip()
    imaging = str(r.get("imaging", "")).strip()
    sex = str(r.get("sex", "")).strip()
    age = str(r.get("age", "")).strip()

    ihc_str = "; ".join([f"{k}:{v}" for k, v in (extra_ihc or {}).items()])
    ish_str = "; ".join([f"{k}:{v}" for k, v in (extra_ish or {}).items()])
    mol_str = "; ".join([f"{k}:{v}" for k, v in (extra_molecular or {}).items()])

    lines = []
    if sex or age:
        lines.append(f"Patient sex: {sex}; age: {age}.")
    if chief_complaint:
        lines.append(f"Chief complaint: {chief_complaint}")
    if history_filtered:
        lines.append(f"History of present illness: {history_filtered}")
    if imaging:
        lines.append(f"Imaging findings: {imaging}")

    return "\n".join(lines)


def to_json_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_clinical_info_record(case_id, case_info):
    return {
        "case_id": str(case_id).strip(),
        "category": to_json_value(case_info.get("category", "")),
        "sex": to_json_value(case_info.get("sex", "")),
        "age": to_json_value(case_info.get("age", "")),
        "chief_complaint": to_json_value(case_info.get("chief_complaint", "")),
        "history_after_removing_pathology_information": to_json_value(
            case_info.get("history_after_removing_pathology_information", "")
        ),
        "imaging": to_json_value(case_info.get("imaging", "")),
        "pathology": to_json_value(case_info.get("pathology", "")),
    }


def save_clinical_info(case_id, case_info):
    clinical_record = build_clinical_info_record(case_id, case_info)
    clinical_path = os.path.join(clinical_info_dir, f"{case_id}.json")
    with open(clinical_path, "w", encoding="utf-8") as f:
        json.dump(clinical_record, f, ensure_ascii=False, indent=2)
    print(f"[OK] Clinical information saved: {clinical_path}")
    return clinical_path


# ============================================================
#                    Initialize DeepSeek Reasoner
# ============================================================
client = OpenAI(
    api_key="YOUR-KEY",
    base_url="https://api.deepseek.com"
)


def build_rag_prompt(case_id, clinical_query):
    return f"""
    You are an expert hematopathologist with extensive experience in lymphoma diagnosis.
    
    Based solely on the available clinical information, perform an initial differential diagnosis.
    
    Purpose of this stage:
    
    This represents the initial clinical differential diagnosis phase prior to histopathological confirmation.
    
    The objectives are:
    
    1. Exclude lymphoma entities that are clinically incompatible with the available presentation.
    2. Identify the most plausible diagnostic candidates requiring pathological, immunophenotypic, and molecular confirmation.
    3. Guide subsequent diagnostic workup and ancillary testing.
    
    Diagnostic Considerations:
    
    - Patient age and sex
    - Presenting symptoms
    - Disease distribution and involved anatomical sites
    - Disease aggressiveness and progression pattern
    - Radiological findings
    - Typical epidemiological and clinical characteristics of each lymphoma subtype
    
    The assigned probabilities should reflect the likelihood of each diagnosis based solely on currently available clinical information and should not be interpreted as final diagnoses.
    
    --------------------------------------------------
    
    Patient Information:
    
    {clinical_query}
    
    --------------------------------------------------
    
    Return ONLY valid JSON.
    
    Do not provide explanations outside the JSON structure.
    
    {{
      "case_id": "{case_id}",
      "predictions": [
        {{"name": "Burkitt Lymphoma", "probability": 0.00}},
        {{"name": "B-Lymphoblastic Leukemia/Lymphoma", "probability": 0.00}},
        {{"name": "T-Lymphoblastic Lymphoma", "probability": 0.00}},
        {{"name": "High-Grade B-Cell Lymphoma with MYC and BCL2 Rearrangements", "probability": 0.00}},
        {{"name": "ALK-Positive Anaplastic Large Cell Lymphoma", "probability": 0.00}},
        {{"name": "ALK-Negative Anaplastic Large Cell Lymphoma", "probability": 0.00}},
        {{"name": "Plasmacytoma", "probability": 0.00}},
        {{"name": "Nodular Lymphocyte-Predominant Hodgkin Lymphoma", "probability": 0.00}},
        {{"name": "Nodal Marginal Zone Lymphoma", "probability": 0.00}},
        {{"name": "Extranodal NK/T-Cell Lymphoma", "probability": 0.00}},
        {{"name": "Classical Hodgkin Lymphoma, Mixed Cellularity Type", "probability": 0.00}},
        {{"name": "Classical Hodgkin Lymphoma, Nodular Sclerosis Type", "probability": 0.00}},
        {{"name": "Classical Hodgkin Lymphoma, Lymphocyte-Rich Type", "probability": 0.00}},
        {{"name": "Follicular Lymphoma", "probability": 0.00}},
        {{"name": "Chronic Lymphocytic Leukemia / Small Lymphocytic Lymphoma", "probability": 0.00}},
        {{"name": "Diffuse Large B-Cell Lymphoma, NOS", "probability": 0.00}},
        {{"name": "Mantle Cell Lymphoma", "probability": 0.00}},
        {{"name": "Peripheral T-Cell Lymphoma, NOS", "probability": 0.00}},
        {{"name": "Angioimmunoblastic T-Cell Lymphoma", "probability": 0.00}},
        {{"name": "MALT Lymphoma", "probability": 0.00}}
      ],
    
      "clinical_analysis":
      "Provide a comprehensive analysis of the patient's demographic characteristics, clinical manifestations, disease distribution, and radiological findings, and discuss their compatibility with each lymphoma subtype.",
    
      "exclusion_reasoning":
      "Explain why certain lymphoma entities are considered unlikely based on the available clinical evidence.",
    
      "candidate_justification":
      "Provide supporting clinical evidence for the leading diagnostic candidates.",
    
      "most_likely":
      ["Diagnosis 1", "Diagnosis 2", "Diagnosis 3", "Diagnosis 4", "Diagnosis 5"],
    
      "least_likely":
      ["Diagnosis 1", "Diagnosis 2", "Diagnosis 3", "Diagnosis 4", "Diagnosis 5"],
    
      "next_step_recommendation":
      "Recommend the most appropriate pathological, immunohistochemical, flow cytometric, cytogenetic, and molecular studies required for definitive diagnosis."
    }}
    """


def call_deepseek(rag_prompt):
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a hematopathology expert. Perform a preliminary lymphoma differential diagnosis based on clinical information and provide direction for subsequent pathologic evaluation."},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.2,
        max_tokens=7000
    )
    return response.choices[0].message.content


def run_prehe_agent(case_info, save_result=True, output_path=None):
    """Run the first agent. case_info can come from Excel or interactive input."""
    case_id = str(case_info.get("case_id", "")).strip()
    if not case_id:
        case_id = f"interactive_{int(time.time())}"

    clinical_info_path = None
    if save_result:
        clinical_info_path = save_clinical_info(case_id, case_info)

    clinical_query = build_clinical_query(
        case_id,
        {},  # only H&E diagnosis
        case_info,
        {},
        {},
        {}
    )

    rag_prompt = build_rag_prompt(case_id, clinical_query)

    try:
        reply_text = call_deepseek(rag_prompt)
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
                "clinical_info_path": clinical_info_path,
                "rag_prompt_used": rag_prompt,
                "rag_reply_raw": reply_text,
                "rag_parsed_json": None,
            }

    except Exception as e:
        print(f"[X] DeepSeek call failed: {e}")
        return None

    result = {
        "case_id": case_id,
        "clinical_info_path": clinical_info_path,
        "rag_prompt_used": rag_prompt,
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


def read_multiline(prompt_text):
    print(prompt_text)
    print("Press Enter on an empty line to finish. For multi-line input, enter one line at a time and type END when finished.")
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


def collect_case_info_from_dialog():
    print("\n=== CTA: Interactive Clinical Information Input ===")
    print("Enter the case information step by step. Press Enter to skip unavailable items.\n")

    case_info = {
        "case_id": input("Pathology number / case ID: ").strip(),
        "category": "",
        "sex": input("Sex: ").strip(),
        "age": input("Age: ").strip(),
        "chief_complaint": read_multiline("Chief complaint:"),
        "history_after_removing_pathology_information": read_multiline("History of present illness after removing pathology information:"),
        "imaging": read_multiline("Imaging report / imaging summary:"),
        "pathology": "",
    }

    print("\n--- Input Summary ---")
    print(build_clinical_query(case_info.get("case_id", ""), {}, case_info, {}, {}, {}))
    confirm = input("\nCall DeepSeek now? Enter y to continue, or any other key to cancel: ").strip().lower()
    if confirm != "y":
        print("Canceled.")
        return None
    return case_info


def run_interactive():
    while True:
        case_info = collect_case_info_from_dialog()
        if case_info:
            case_id = str(case_info.get("case_id", "")).strip()
            if not case_id:
                case_id = f"interactive_{int(time.time())}"
                case_info["case_id"] = case_id
            print(f"\n=== Processing case: {case_id} ===")
            result = run_prehe_agent(case_info)
            if result and result.get("rag_parsed_json") is not None:
                print("\n--- DeepSeek JSON Result ---")
                print(json.dumps(result["rag_parsed_json"], ensure_ascii=False, indent=2))

        again = input("\nEnter another case? Enter y to continue, or any other key to exit: ").strip().lower()
        if again != "y":
            break

    print("\n=== Interactive input finished ===")


def load_test_dataframe(excel_path):
    test_df = pd.read_excel(excel_path, header=None)
    test_df.columns = [
        "case_id",
        "category",
        "chief_complaint",
        "history_after_removing_pathology_information",
        "imaging",
        "sex",
        "age",
        "pathology",
    ]
    return test_df


def run_batch(excel_path=path):
    test_df = load_test_dataframe(excel_path)
    print(test_df.head())

    clinical_bank = {}
    for _, r in test_df.iterrows():
        case_id = str(r["case_id"]).strip()
        if case_id and case_id.lower() != "nan":
            clinical_bank[case_id] = r

    for _, row in test_df.iterrows():
        case_id = str(row.iloc[0]).strip()

        if case_id in ["case_id", "", "nan"]:
            continue

        print(f"\n=== Processing case: {case_id} ===")

        output_path = os.path.join(output_dir, f"{case_id}.json")
        if os.path.exists(output_path):
            print("Already generated. Skipping.")
            continue

        r = clinical_bank.get(case_id, {})
        run_prehe_agent(r, save_result=True, output_path=output_path)
        time.sleep(1)

    print("\n=== All processing completed ===")


def parse_args():
    parser = argparse.ArgumentParser(description="First agent: preliminary differential diagnosis based on chief complaint, history, and imaging.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start the interactive input window")
    parser.add_argument("--excel", default=path, help="Excel path used in batch mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.interactive:
        run_interactive()
    else:
        run_batch(args.excel)
