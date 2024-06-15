# Machine Translation for Biomedical Text (inc. HPO)
English-Spanish machine translation system for biomedical text and HPO terminology


## Translation of free text

```bash
python translation/text/main.py <input_file> <output_file> --evaluate --preexpansion --postexpansion --synonyms
```
* `<input_file>`: The file containing the text to be translated. Should be a _.txt_ file with one sentence per line or a _.jsonl_ file with _"en"_ and _"es"_ keys on each line for original English sentence and Spanish reference translation respectively.
* `<output_file>`: The file to store translation results. Should have _.csv_ extension.
* `--evaluate` (optional): Evaluate the translation quality sentence-by-sentence using SacreBLEU, TER and semantic similarity. Can only be used with a _.jsonl_ input file.
* `--preexpansion` (optional): Expand the input English text abbreviations where possible.
* `--postexpansion` (optional): Expand the output Spanish text abbreviations where possible.
* `--synonyms` (optional): Substitute secondary for primary synonyms according to the_ Panhispanic dictionary of medical terms_ in model translations where possible.

### Output
The output CSV file is a table with the following columns:
* _english_: the original English sentence
* _candidate_: the model's Spanish translation
* _reference_: the Spanish reference translation (only if provided in the _.jsonl_ file)

If the `--evaluate` flag is on, the table additionally has these columns:
* _sacrebleu_: sentence-level SacreBLEU score
* _\`ter\`_: sentence-level \`TER\` (1 - Translation Error Rate) 
* _semsim_: sentence-level semantic similarity 

## Translation of HPO terms
```bash
python translation/hpo/main.py <hpo_id> <model_checkpoint>
```
* `<hpo_id>`: The HPO ID translate from. Captures all subterms recursively. Use HP:0000001 to evaluate on all terms. 
* `<model_checkpoint>`: The model checkpoint to use for translation. Should be accessible on Hugging Face and have a MarianMT architecture, otherwise, the code requires editing.

### Output
A CSV file named `<hpo_id>.csv` is created with the following columns:
* _hpo_id_: ID of the HPO term
* _english_: the English label of the term
* _candidate_: the model's Spanish translation of the term
* _reference_: the official Spanish label of the term 
* _simple_: a simple string comparison of the translation and label, 1 if they are identical and 0 otherwise
* _sacrebleu_: sentence-level SacreBLEU score
* _\`ter\`_: sentence-level \`TER\` (1 - Translation Error Rate) 
* _semsim_: sentence-level semantic similarity 
