# Machine Translation for Biomedicine (inc. HPO)
Biomedical domain-specific translation from English to Spanish


## Translation of free text

```bash
python translation/text/main.py <input_file> <output_file> --evaluate --preexpansion --postexpansion --synonyms
```
* `<input_file>`: The file containing the text to be translated. Should be a _.txt_ file with one sentence per line or a _.jsonl_ file with "en" and "es" keys on each line for original English and reference translation respectively.
* `<output_file>`: The file to write the translation table to. Should have _.csv_ extension.
* `--evaluate` (optional): Evaluate the translation quality sentence-by-sentence using SacreBLEU, TER and semantic similarity. Can only be used with a _.jsonl_ input file.
* `--preexpansion` (optional): Expand the input English text abbreviations where possible.
* `--postexpansion` (optional): Expand the output Spanish text abbreviations where possible.
* `--synonyms` (optional): Substitute secondary for primary synonyms in output Spanish text where possible.

### Output
The output CSV file is a table with the following columns:
* _english_: the original English sentence
* _reference_: the reference Spanish translation
* _translation_: the model's Spanish output
* _sacrebleu_: sentence-level SacreBLEU score
* _\`ter\`_: sentence-level TER 
* _semsim_: sentence-level semantic similarity 

## Translation of HPO terms
```bash
python translation/hpo/main.py <hpo_id> <model_checkpoint>
```
* `<hpo_id>`: The HPO ID translate from. Captures all subterms recursively. Use HP:0000001 to evaluate on all terms. 
* `<model_checkpoint>`: The model checkpoint to use for translation. Should be accessible on Hugging Face and have a MarianMT architecture, otherwise the code must be minorly edited.

### Output
In the _results/_ directory, a .XLSX and CSV named after the HPO code are created.
The CSV contains the following columns:
* _hpo_id_: ID of the HPO term
* _english_: the English label of the term
* _traducción modelo_: the model's Spanish translation of the term
* _etiqueta oficial_: the official Spanish label of the term
* _SIMPLE_: a simple string comparison of the translation and label, 1 if they are identical and 0 otherwise
* _SACREBLEU_: sentence-level SacreBLEU score 
* _TER_: sentence-level TER 
* _SEMSIM_: semantic similarity of the translation using the model sentence-transformers
/paraphrase-multilingual-mpnet-base-v2
