# HPO Evaluator

## Description

Evaluates the English-Spanish translation capabilities of the model. Uses the HPO official translations as a benchmark

## Prerequisites

- Python 3.11

## Usage

```bash
python evaluation/main.py <hpo_id> --xlsx <filename> --labels
```
* <hpo_id>: The HPO ID to begin the evaluation from. Use HP:0000001 to evaluate on all terms. Captures all subclass terms.
* --xlsx <filename> (Optional): The name of the xlsx file to read already generated model translations from.
* --labels (Optional): Only evaluate the translations of term labels, not descriptions, synonyms etc.


