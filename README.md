# HDLQA: Measuring and Mitigating Over-Answering in LLM-Based HDL Question Answering

This repository contains the dataset and source code for the paper *"When LLMs Over-Answer: Measuring and Mitigating Quality Issues in LLM-Based Hardware Description Language Question Answering."*

## Repository Structure
.
├── code/
│   ├── clean_tags/clean_tags_V3.py   # Keyword-based filtering of HDL-related Stack Overflow posts
│   ├── dataset/llm_class.py          # LLM-based task-type classification (taxonomy labeling)
│   └── multi-agent/
│       ├── pipeline.py               # Multi-agent refinement framework (M1 generation, M2 refinement, M3 judge)
│       └── m5_evaluation.py          # Evaluation: quality scores, core number/length, CC/CP/DP
│
└── VerilogQA(dataset)/
├── classified_verilog-full.json  # Full dataset: 6,246 HDL Q&A pairs with accepted answers and task labels
└── sample_363.json               # Stratified 363-question sample used in the user study and evaluation

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
