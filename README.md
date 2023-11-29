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
* --labels (Optional): Only evaluate the translations of term labels, not descriptions, synonyms etc.


## Output
In the /files/results/official directory, a CSV named after the HPO code is created. The CSV contains the following columns:
* HPO ID: the ID of the term
* Kind: the type of term translation (label, description, synonym etc.)
* English: the English label of the term
* Traducción modelo: the translation of the term by the model
* Etiqueta oficial: the official HPO translation of the term
* Columns for each similarity metric