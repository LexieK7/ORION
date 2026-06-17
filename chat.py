import json

import CTA
import IPA
import MWPA
import IDA


def read_multiline(prompt_text):
    print(prompt_text)
    print("Enter one or more lines. Type END when finished. Press Enter on an empty line to skip.")
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


def collect_clinical_info(case_id):
    print("\n=== CTA: Clinical Information ===")
    print("Enter the clinical information for this case. Press Enter to skip unavailable items.\n")
    return {
        "case_id": case_id,
        "category": "",
        "sex": input("Sex: ").strip(),
        "age": input("Age: ").strip(),
        "chief_complaint": read_multiline("Chief complaint:"),
        "history_after_removing_pathology_information": read_multiline(
            "History of present illness:"
        ),
        "imaging": read_multiline("Imaging report / imaging summary:"),
        "pathology": "",
    }


def collect_he_predictions():
    print("\n=== H&E Image Model Prediction ===")
    text = read_multiline(
        "Enter H&E image model prediction(s). Examples: 'Diffuse large B-cell lymphoma, not otherwise specified: 0.82' or a JSON list."
    )
    return IPA.parse_he_predictions(text)


def get_parsed(result):
    if not result:
        return None
    return result.get("rag_parsed_json")


def print_prediction_list(predictions, title="Predictions", limit=10):
    print(f"\n--- {title} ---")
    predictions = predictions or []
    if not predictions:
        print("(No predictions available.)")
        return
    for idx, item in enumerate(predictions[:limit], start=1):
        if not isinstance(item, dict):
            print(f"{idx}. {item}")
            continue
        name = item.get("name", "")
        probability = item.get("probability", None)
        try:
            print(f"{idx}. {name}: {float(probability):.4f}")
        except:
            print(f"{idx}. {name}")


def print_cta_summary(result):
    parsed = get_parsed(result)
    print("\n================ CTA RESULT ================")
    if not parsed:
        print("CTA did not return valid parsed JSON.")
        return
    print_prediction_list(parsed.get("predictions", []), "CTA top predictions")
    if parsed.get("clinical_analysis"):
        print("\nClinical analysis:")
        print(parsed.get("clinical_analysis"))
    if parsed.get("next_step_recommendation"):
        print("\nNext step recommendation:")
        print(parsed.get("next_step_recommendation"))


def print_ipa_summary(result):
    parsed = get_parsed(result)
    print("\n================ IPA RESULT ================")
    if not parsed:
        print("IPA did not return valid parsed JSON.")
        return
    subtypes = parsed.get("most_confusing_subtypes", [])
    if subtypes:
        print("\nMost confusing subtypes:")
        for subtype in subtypes:
            print(f"- {subtype}")
    if parsed.get("key_differential_features"):
        print("\nKey differential features:")
        print(parsed.get("key_differential_features"))
    final_panel = parsed.get("final_IHC_panel", [])
    print("\nFinal IHC panel recommended by IPA:")
    if final_panel:
        for marker in final_panel:
            print(f"- {marker}")
    else:
        print("(No final_IHC_panel returned.)")


def print_mwpa_summary(result):
    parsed = get_parsed(result)
    print("\n================ MWPA RESULT ================")
    if not parsed:
        print("MWPA did not return valid parsed JSON.")
        return None
    print_prediction_list(parsed.get("final_prediction", []), "MWPA integrated predictions")
    if parsed.get("key_evidence"):
        print("\nKey evidence:")
        print(parsed.get("key_evidence"))
    if parsed.get("diagnostic_conclusion"):
        print("\nDiagnostic conclusion:")
        print(parsed.get("diagnostic_conclusion"))
    certainty = str(parsed.get("diagnostic_certainty", "")).strip().lower()
    print(f"\nDiagnostic certainty: {certainty or '(not provided)'}")
    tests = parsed.get("suggested_next_tests", [])
    if tests:
        print("\nSuggested next tests:")
        for test in tests:
            print(f"- {test}")
    return certainty


def print_ida_summary(result):
    parsed = get_parsed(result)
    print("\n================ IDA FINAL RESULT ================")
    if not parsed:
        print("IDA did not return valid parsed JSON.")
        return
    print_prediction_list(parsed.get("final_prediction", []), "Final diagnostic predictions")
    if parsed.get("key_evidence"):
        print("\nFinal key evidence:")
        print(parsed.get("key_evidence"))
    if parsed.get("diagnostic_conclusion"):
        print("\nFinal diagnostic conclusion:")
        print(parsed.get("diagnostic_conclusion"))
    print(f"\nDiagnostic certainty: {parsed.get('diagnostic_certainty', '(not provided)')}")
    tests = parsed.get("suggested_next_tests", [])
    if tests:
        print("\nAdditional suggested tests:")
        for test in tests:
            print(f"- {test}")


def run_case():
    print("\n############################################")
    print("Integrated Lymphoma Agent Workflow")
    print("CTA -> H&E input -> IPA -> MWPA -> IDA if needed")
    print("############################################")

    case_id = input("\nPathology number / case ID: ").strip()
    if not case_id:
        print("Case ID is required.")
        return

    clinical_info = collect_clinical_info(case_id)

    print(f"\n>>> Running CTA for case {case_id} ...")
    cta_result = CTA.run_prehe_agent(clinical_info)
    print_cta_summary(cta_result)
    if not get_parsed(cta_result):
        print("\nWorkflow stopped because CTA did not produce valid parsed JSON.")
        return

    he_predictions = collect_he_predictions()
    if not he_predictions:
        print("\nWorkflow stopped because no valid H&E prediction was provided.")
        return
    print_prediction_list(he_predictions, "Entered H&E predictions")

    print(f"\n>>> Running IPA for case {case_id} ...")
    ipa_cta = IPA.load_cta_result(case_id)
    ipa_result = IPA.run_ipa_agent(
        case_id=case_id,
        clinical_info=ipa_cta["clinical_text"],
        cta_predictions=ipa_cta["cta_predictions"],
        he_predictions=he_predictions,
    )
    print_ipa_summary(ipa_result)
    if not get_parsed(ipa_result):
        print("\nWorkflow stopped because IPA did not produce valid parsed JSON.")
        return

    mwpa_context = MWPA.load_upstream_context(case_id)
    final_ihc_panel = MWPA.normalize_ihc_panel(mwpa_context.get("final_ihc_panel", []))

    print("\n=== IHC Results for MWPA ===")
    if final_ihc_panel:
        print("IPA recommended the following IHC panel for this case:")
        for marker in final_ihc_panel:
            print(f"- {marker}")
    else:
        print("No IPA final_IHC_panel was found. You may enter IHC results manually.")
    ihc_results = MWPA.collect_ihc_results_from_dialog(final_ihc_panel)

    print(f"\n>>> Running MWPA for case {case_id} ...")
    mwpa_result = MWPA.run_mwpa_agent(
        case_id=case_id,
        upstream_context=mwpa_context,
        ihc_results=ihc_results,
    )
    certainty = print_mwpa_summary(mwpa_result)
    if not get_parsed(mwpa_result):
        print("\nWorkflow stopped because MWPA did not produce valid parsed JSON.")
        return

    if certainty == "high":
        print("\nMWPA returned high diagnostic certainty. IDA will not be called.")
        print("\nFinal result is the MWPA result above.")
        return

    ida_context = IDA.load_upstream_context(case_id)
    suggested_tests = IDA.normalize_test_panel(ida_context.get("suggested_next_tests", []))

    print("\n=== Molecular / Cytogenetic / ISH Results for IDA ===")
    if suggested_tests:
        print("MWPA recommended the following next tests:")
        for test in suggested_tests:
            print(f"- {test}")
    else:
        print("No MWPA suggested_next_tests were found. You may enter completed ancillary results manually.")
    molecular_results = IDA.collect_molecular_results_from_dialog(suggested_tests)

    print(f"\n>>> Running IDA for case {case_id} ...")
    ida_result = IDA.run_ida_agent(
        case_id=case_id,
        upstream_context=ida_context,
        molecular_results=molecular_results,
    )
    print_ida_summary(ida_result)


def main():
    while True:
        run_case()
        again = input("\nRun another case? Enter y to continue, or any other key to exit: ").strip().lower()
        if again != "y":
            break
    print("\nIntegrated workflow finished.")


if __name__ == "__main__":
    main()
