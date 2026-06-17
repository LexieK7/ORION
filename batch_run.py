import argparse
import os
import shutil

import CTA
import IPA
import MWPA
import IDA


DEFAULT_CLINICAL_EXCEL = "./clinical_cases.xlsx"
DEFAULT_HE_EXCEL = "./he_predictions.xlsx"
DEFAULT_IHC_EXCEL = "./ihc_results.xlsx"
DEFAULT_MOLECULAR_EXCEL = "./molecular_results.xlsx"

DEFAULT_CTA_DIR = "./outputs/cta_results"
DEFAULT_IPA_DIR = "./outputs/ipa_results"
DEFAULT_MWPA_DIR = "./outputs/mwpa_results"
DEFAULT_IDA_DIR = "./outputs/ida_results"

STAGES = ["cta", "ipa", "mwpa", "ida"]


def normalize_stage_list(raw_value):
    if not raw_value:
        return STAGES
    stages = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    invalid = [stage for stage in stages if stage not in STAGES]
    if invalid:
        raise ValueError(f"Unknown stage(s): {', '.join(invalid)}. Valid stages: {', '.join(STAGES)}")
    return stages


def clean_dir(path):
    if not path:
        return
    abs_path = os.path.abspath(path)
    cwd = os.path.abspath(os.getcwd())
    if not abs_path.startswith(cwd):
        raise ValueError(f"Refusing to delete outside the current workspace: {abs_path}")
    if os.path.isdir(abs_path):
        shutil.rmtree(abs_path)
    os.makedirs(abs_path, exist_ok=True)


def clean_outputs_for_stages(stages, args):
    if "cta" in stages:
        clean_dir(args.cta_dir)
    if "ipa" in stages:
        clean_dir(args.ipa_dir)
    if "mwpa" in stages:
        clean_dir(args.mwpa_dir)
    if "ida" in stages:
        clean_dir(args.ida_dir)


def ensure_input_files(args, stages):
    required = []
    if any(stage in stages for stage in ["cta", "ipa", "mwpa", "ida"]):
        required.append(args.clinical_excel)
    if any(stage in stages for stage in ["ipa", "mwpa", "ida"]):
        required.append(args.he_excel)
    if "mwpa" in stages:
        required.append(args.ihc_excel)
    if "ida" in stages:
        required.append(args.molecular_excel)

    missing = [path for path in required if path and not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))


def run_cta(args):
    print("\n==============================")
    print("Running CTA batch")
    print("==============================")
    CTA.run_batch(args.clinical_excel)


def run_ipa(args):
    print("\n==============================")
    print("Running IPA batch")
    print("==============================")
    IPA.run_batch(
        clinical_excel=args.clinical_excel,
        he_excel=args.he_excel,
        cta_dir=args.cta_dir,
        cta_excel=args.cta_excel,
    )


def run_mwpa(args):
    print("\n==============================")
    print("Running MWPA batch")
    print("==============================")
    MWPA.run_batch(
        clinical_excel=args.clinical_excel,
        he_excel=args.he_excel,
        cta_dir=args.cta_dir,
        ipa_dir=args.ipa_dir,
        ihc_excel=args.ihc_excel,
        cta_excel=args.cta_excel,
    )


def run_ida(args):
    print("\n==============================")
    print("Running IDA batch")
    print("==============================")
    IDA.run_batch(
        clinical_excel=args.clinical_excel,
        he_excel=args.he_excel,
        cta_dir=args.cta_dir,
        ipa_dir=args.ipa_dir,
        mwpa_dir=args.mwpa_dir,
        molecular_excel=args.molecular_excel,
        cta_excel=args.cta_excel,
        mwpa_excel=args.mwpa_excel,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch Excel tests across CTA, IPA, MWPA, and IDA in sequence."
    )
    parser.add_argument("--clinical-excel", default=DEFAULT_CLINICAL_EXCEL)
    parser.add_argument("--he-excel", default=DEFAULT_HE_EXCEL)
    parser.add_argument("--ihc-excel", default=DEFAULT_IHC_EXCEL)
    parser.add_argument("--molecular-excel", default=DEFAULT_MOLECULAR_EXCEL)
    parser.add_argument("--cta-excel", default=None, help="Optional CTA prediction Excel fallback for downstream stages")
    parser.add_argument("--mwpa-excel", default=None, help="Optional MWPA prediction Excel fallback for IDA")
    parser.add_argument("--cta-dir", default=DEFAULT_CTA_DIR)
    parser.add_argument("--ipa-dir", default=DEFAULT_IPA_DIR)
    parser.add_argument("--mwpa-dir", default=DEFAULT_MWPA_DIR)
    parser.add_argument("--ida-dir", default=DEFAULT_IDA_DIR)
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help="Comma-separated stages to run. Valid values: cta,ipa,mwpa,ida",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output directories for selected stages before running so existing JSON files do not cause skips.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stages = normalize_stage_list(args.stages)

    ensure_input_files(args, stages)

    if args.force:
        clean_outputs_for_stages(stages, args)

    runners = {
        "cta": run_cta,
        "ipa": run_ipa,
        "mwpa": run_mwpa,
        "ida": run_ida,
    }

    for stage in stages:
        runners[stage](args)

    print("\n==============================")
    print("Batch test completed")
    print("==============================")


if __name__ == "__main__":
    main()
