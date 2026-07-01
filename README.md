# HDLQA: Measuring and Mitigating Over-Answering in LLM-Based HDL Question Answering

This repository contains the dataset and source code for the paper *"When LLMs Over-Answer: Measuring and Mitigating Quality Issues in LLM-Based Hardware Description Language Question Answering."*

## Repository Structure
- **Root directory**: Contains the dataset files (`classified_verilog-full.json` and `sample_363.json`), this `README.md`, and any necessary environment configuration files (if applicable).
- **`code/`**: Main directory for all source code, organised into three submodules:
  - `clean_tags/` — data cleaning and filtering scripts (e.g., `clean_tags_V3.py`);
  - `dataset/` — task‑type classification scripts (e.g., `llm_class.py`);
  - `multi-agent/` — complete pipeline for the multi‑agent framework (e.g., `pipeline.py`).
- **Other directories** (e.g., `results/` or `logs/`) may be created automatically during execution to store outputs and intermediate files.


## Dataset

- **classified_verilog-full.json** — 6,246 real-world HDL Q&A pairs mined from the public Stack Overflow data dump. Each record contains the question (title and body), the human-accepted answer, tags, and a task-type label (four main categories, ten subcategories).
- **sample_363.json** — the stratified random sample (n = 363) used in the user study and framework evaluation, preserving the category distribution of the full dataset.


## Usage

1. **Filtering** — build the HDL-related post set:
```bash
   python code/clean_tags/clean_tags_V3.py
```
2. **Classification** — assign task-type labels:
```bash
   python code/dataset/llm_class.py
```
3. **Refinement** — run the multi-agent framework on a question set:
```bash
   python code/multi-agent/pipeline.py
```


## Notes

- All LLM stages run at temperature 0.
- The framework supports multiple backbone models; the LLM-as-Judge uses a separate model from the answer model.

## License

Released for research purposes to facilitate further work on HDL question answering.
