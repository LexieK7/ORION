# ORION

**Multi-agent collaborative reasoning enables interpretable and adaptive clinical diagnosis**

ORION is an **autOnomous, inteRpretable, multI-agent, multimOdal diagNostic system** designed to model disease diagnosis as a sequential process of collaborative clinical reasoning rather than a single static prediction task. The system coordinates specialized agents for clinical interpretation, diagnostic planning, immunohistochemistry recommendation, molecular testing recommendation, and final multimodal evidence integration.

This repository contains the open-source workflow code for the ORION agents.


## Quick Links

- 🧭 **Full interactive workflow:** `python chat.py`
- 📊 **Batch Excel workflow:** `python batch_run.py`
- 🧬 **Agents:** Clinical triage agent (CTA), Context-aware pathology agent (CPA), Immunophenotyping planning agent (IPA), Molecular workup planning agent (MWPA), Integrative diagnostic agent (IDA)


## Overview

Clinical diagnosis requires progressive integration of heterogeneous evidence, including clinical presentation, imaging, H&E morphology, immunophenotype, molecular testing, and clinician feedback. ORION implements this process as a traceable multi-agent workflow in which each agent contributes a structured diagnostic reasoning step and writes its outputs to JSON files that can be inspected, reused, or corrected by clinicians.

In our study, ORION was evaluated on multicenter retrospective and prospective real-world cohorts. By progressively integrating multimodal evidence and clinician-guided interaction, ORION improved diagnostic performance across sequential diagnostic stages and enabled transparent human-AI collaboration.

## Important Note About This Release

This public version **does not include local historical case retrieval or literature retrieval**.

These components were intentionally removed because:

- Historical case banks may contain sensitive patient-related information.
- Literature and reference databases may include licensed or authorization-restricted resources.

Therefore, this release focuses on the deployable multi-agent reasoning workflow without private case-bank retrieval or copyrighted literature retrieval.


## Repository Contents

| File | Agent | Role |
|---|---|---|
| `CTA.py` | Clinical Text Agent | Performs preliminary differential diagnosis from clinical information. |
| `CPA` | Context-aware pathology agent | H&E image analysis. |
| `IPA.py` | Immunophenotyping Planning Agent | Recommends a case-specific IHC panel based on CTA and CPA predictions. |
| `MWPA.py` | Molecular Workup Planning Agent | Integrates CTA, CPA, IPA, and completed IHC results, then recommends molecular, cytogenetic, pathogen-associated, or ISH tests when needed. |
| `IDA.py` | Integrated Diagnosis Agent | Produces the final integrated diagnostic impression from all available upstream evidence. |
| `chat.py` | Interactive Orchestrator | Runs the full workflow interactively by case ID. |
| `batch_test.py` | Batch Test Orchestrator | Runs all agents sequentially using Excel input files. |
| `clinical_cases.xlsx` | Example data | Two demo clinical cases for batch testing. |
| `he_predictions.xlsx` | Example data | Two demo H&E image model prediction vectors for batch testing. |
| `ihc_results.xlsx` | Example data | Two demo IHC result records for batch testing. |
| `molecular_results.xlsx` | Example data | Two demo molecular/ISH result records for batch testing. |

**It should be noted that the "Example" mentioned here is not real; it is a randomly generated example by the AI for testing purposes.**

## Workflow

The complete ORION workflow in this repository is:

```text
CTA -> CPA -> IPA -> MWPA -> IDA
```

1. **CTA** receives clinical information and imaging summaries.
2. **CPA** provides H&E prediction results.
3. **IPA** recommends an IHC panel based on CTA and CPA predictions.
4. **MWPA** reads the IPA-recommended IHC panel and accepts completed IHC results.
5. **IDA** reads MWPA-recommended molecular/ISH tests and accepts completed molecular results for final diagnosis.

If MWPA returns `diagnostic_certainty = "high"`, the interactive workflow stops before IDA because additional molecular testing is not required for the final diagnostic step.

## Installation

Create a Python environment and install dependencies:

```bash
pip install pandas numpy openai openpyxl
```

Recommended Python version:

```text
Python >= 3.10
```

## LLM Configuration

The current scripts use the OpenAI-compatible DeepSeek API client:

```python
client = OpenAI(
    api_key="YOUR_KEY",
    base_url="https://api.deepseek.com"
)
```

Before running the system, replace the placeholder or existing API key in:

- `CTA.py`
- `IPA.py`
- `MWPA.py`
- `IDA.py`

You can deploy ORION with any advanced LLM that supports an OpenAI-compatible chat completion API. To switch models or providers, modify:

```python
client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="YOUR_PROVIDER_BASE_URL"
)
```

and the model name in each `call_deepseek(...)` function:

```python
response = client.chat.completions.create(
    model="deepseek-reasoner",
    ...
)
```

For example, replace `"deepseek-reasoner"` with the model name required by your LLM provider.

## Interactive Full Workflow

Run the integrated interactive interface:

```bash
python chat.py
```

The interface will ask for:

1. Case ID.
2. Clinical information for CTA.
3. H&E image model prediction results from your external CPA.
4. IHC results for the IPA-recommended `final_IHC_panel`.
5. Molecular/ISH/cytogenetic results for the MWPA-recommended `suggested_next_tests`, if MWPA certainty is not high.

All intermediate outputs are saved automatically under `outputs/`.

## Running Agents Individually

Each agent can also be run interactively.

### 🩺 CTA

```bash
python CTA.py --interactive
```

User inputs:

- Case ID
- Sex
- Age
- Chief complaint
- History of present illness
- Imaging report or summary

Outputs:

```text
outputs/cta_results/{case_id}.json
outputs/cta_results/clinical_info/{case_id}.json
```

### 🧫 IPA

```bash
python IPA.py --interactive
```

IPA reads CTA output by case ID and asks the user to input H&E model predictions.

Accepted H&E input examples:

```text
Diffuse large B-cell lymphoma, not otherwise specified: 0.82
Follicular lymphoma: 0.11
```

or a JSON list:

```json
[
  {"name": "Diffuse large B-cell lymphoma, not otherwise specified", "probability": 0.82},
  {"name": "Follicular lymphoma", "probability": 0.11}
]
```

Outputs:

```text
outputs/ipa_results/{case_id}.json
outputs/ipa_results/he_predictions/{case_id}.json
```

### 🧬 MWPA

```bash
python MWPA.py --interactive
```

MWPA reads CTA, H&E, and IPA outputs by case ID. It displays the IPA-recommended `final_IHC_panel` and asks the user to enter completed IHC results marker by marker.

If the user leaves all IHC results empty, the case is treated as:

```text
IHC not performed
```

Outputs:

```text
outputs/mwpa_results/{case_id}.json
outputs/mwpa_results/ihc_results/{case_id}.json
```

### 🧾 IDA

```bash
python IDA.py --interactive
```

IDA reads CTA, H&E, IPA, MWPA, and completed IHC results by case ID. It displays MWPA-recommended `suggested_next_tests` and asks the user to enter completed molecular, cytogenetic, pathogen-associated, or ISH results.

If the user leaves all results empty, the case is treated as:

```text
Molecular/ISH testing not performed
```

Outputs:

```text
outputs/ida_results/{case_id}.json
outputs/ida_results/molecular_results/{case_id}.json
```

## Batch Testing With Excel Files

This repository includes two demo cases for batch testing:

```text
clinical_cases.xlsx
he_predictions.xlsx
ihc_results.xlsx
molecular_results.xlsx
```

Run the full batch workflow:

```bash
python batch_run.py
```

Force rerun by deleting selected output folders first:

```bash
python batch_test.py --force
```

Run only selected stages:

```bash
python batch_test.py --stages cta,ipa
python batch_test.py --stages mwpa,ida
```

Use custom Excel files:

```bash
python batch_test.py \
  --clinical-excel clinical_cases.xlsx \
  --he-excel he_predictions.xlsx \
  --ihc-excel ihc_results.xlsx \
  --molecular-excel molecular_results.xlsx
```

## Expected Excel Formats

### 📋 `clinical_cases.xlsx`

| case_id | category | chief_complaint | history_after_removing_pathology_information | imaging | sex | age | pathology |
|---|---|---|---|---|---|---|---|

### 🖼️ `he_predictions.xlsx`

Two formats are supported.

Format 1:

| Patient_ID | Probabilities |
|---|---|

`Probabilities` should be a JSON-style list of probabilities matching the internal lymphoma label order.

Format 2:

| case_id | lymphoma_entity_1 | lymphoma_entity_2 | ... |
|---|---|---|---|

### 🧫 `ihc_results.xlsx`

| case_id | ihc_results |
|---|---|

Example:

```text
CD20: positive; PAX5: positive; CD3: negative; Ki-67: approximately 80%
```

### 🧬 `molecular_results.xlsx`

| case_id | molecular_results |
|---|---|

Example:

```text
EBER in situ hybridization: negative; MYC rearrangement FISH: negative
```

## Output Structure

```text
outputs/
  cta_results/
    {case_id}.json
    clinical_info/
      {case_id}.json
  ipa_results/
    {case_id}.json
    he_predictions/
      {case_id}.json
  mwpa_results/
    {case_id}.json
    ihc_results/
      {case_id}.json
  ida_results/
    {case_id}.json
    molecular_results/
      {case_id}.json
```

Each JSON file stores:

- The case ID
- The prompt used
- The raw LLM response
- The parsed JSON output
- Paths to upstream evidence when applicable

This makes the diagnostic chain traceable and auditable.

## CPA
Please refer to the CPA repository.



## Citation

If you use ORION in your research, please cite:

```bibtex
@article{orion2026,
  title   = {Multi-agent collaborative reasoning enables interpretable and adaptive clinical diagnosis},
  author  = {},
  journal = {},
  year    = {2026},
  note    = {}
}
```

Please replace the placeholder author, journal, and publication information with the official citation once available.

## Disclaimer

ORION is a research system for AI-assisted diagnostic reasoning. It is not a standalone medical device and should not be used as the sole basis for clinical diagnosis. All outputs must be reviewed by qualified clinicians and pathologists in the appropriate clinical context.


