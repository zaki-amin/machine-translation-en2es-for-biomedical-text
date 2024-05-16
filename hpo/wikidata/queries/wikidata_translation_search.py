import sys
from collections import defaultdict

from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm

from helper_functions import create_batches, read_json, save_json_to_file, text_cleaning


def get_results(endpoint_url, query):
    """
    Execute a SPARQL query on the given endpoint URL and return the query results.

    Parameters:
        endpoint_url (str): The URL of the SPARQL endpoint to query.
        query (str): The SPARQL query to execute.

    Returns:
        list: A list of dictionaries containing the query results in the SPARQL response.

    Example Usage:
        endpoint_url = 'https://example.org/sparql'  # Replace with the actual SPARQL endpoint URL
        query = 'SELECT ?item ?label WHERE { ?item rdfs:label ?label } LIMIT 10'  # Replace with your SPARQL query
        results = get_results(endpoint_url, query)
    """
    user_agent = f"WDQS-example Python/{sys.version_info[0]}.{sys.version_info[1]}"
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def process_results(results, wikidata_data, batch_data=None, term_search=False):
    """
    Process the query results and update the wikidata_data dictionary with relevant information.

    Parameters:
        results (list of dict): A list of dictionaries containing the query results.
        wikidata_data (dict): A dictionary to store the processed data.
        term_search (bool, optional): Whether the query was performed using term search.
                                      Defaults to False.

    Example Usage:
        results = [...]  # A list of query results
        wikidata_data = {}  # An empty dictionary to store processed data
        process_results(results, wikidata_data)
    """

    # drop duplicates
    # Set to keep track of unique label_en values
    unique_labels = set()

    # New list to store unique dictionaries
    unique_list = []

    # Loop through the original list of dictionaries
    for result in results:
        label = result["label_en"]["value"]
        # Check if the label_en value is unique or not
        if label not in unique_labels:
            unique_labels.add(label)
            unique_list.append(result)

    for result in unique_list:
        if term_search:
            hpo_label = batch_data[result["term"]["value"]]
        else:
            hpo_label = result["hpo_code"]["value"].replace(":", "_")
        if result["label_es"]["value"] or result["synonyms_en"]["value"]:
            wikidata_data[hpo_label]["term_en"] = result["label_en"]["value"]
            wikidata_data[hpo_label]["term_es"] = result["label_es"]["value"]
            wikidata_data[hpo_label]["url"] = result["item"]["value"]

            synonyms_en = (
                result["synonyms_en"]["value"].split(",")
                if isinstance(result["synonyms_en"]["value"], str)
                else result["synonyms_en"]["value"]
            )

            synonyms_es = result["synonyms_es"]["value"].split(",")
            wikidata_data[hpo_label]["synonyms_en"] = [
                text_cleaning(synonym_en) for synonym_en in synonyms_en
            ]
            synonyms_es = synonyms_es if isinstance(synonyms_es, list) else [synonyms_es]
            wikidata_data[hpo_label]["synonyms_es"] = [
                text_cleaning(synonym_es) for synonym_es in synonyms_es
            ]


def search_wikidata(raw_data, hpo_codes, binding_property, batch_size=100, term_search=False):
    """
    Search Wikidata for HPO codes or terms.

    Parameters:
        raw_data: data to use for hpo or term search.
        hpo_codes (list): A list of HPO codes to search for in Wikidata.
        binding_property (str): The binding property to use in the SPARQL query. It can be either "?label" for term search or "wdt:P3841" for HPO codes search.
        batch_size (int, optional): The batch size for processing HPO codes in batches. Defaults to 100.
        term_search (bool, optional): Whether the search is for terms or HPO codes. Defaults to False.

    Returns:
        list: A list of HPO codes or terms that were not found in Wikidata.(only relevant for hpo search)
    """

    hpo_code_found = []
    hpo_batches = create_batches(hpo_codes, batch_size)
    if term_search:
        description = "Using term search to extract from Wikidata"
    else:
        description = "Using HPO code to extract from Wikidata"
    # Iterate over HPO codes or terms in batches
    for hpo_batch in tqdm(hpo_batches, desc=description):
        if term_search:
            # For term search, build a batch data dictionary
            batch_data = {}
            for code in hpo_batch:
                code = code.replace(":", "_")
                label = raw_data[code]["source_label"]
                batch_data[label] = code

                synonyms = raw_data[code].get("source_synonyms")
                if synonyms:
                    for syn in synonyms:
                        batch_data[text_cleaning(syn)] = code
            doid_values = " ".join([f'"{doid}"@en' for doid in list(batch_data.keys())])
        else:
            # For HPO codes search, simply join the values
            doid_values = " ".join([f'"{doid}"' for doid in hpo_batch])
        # Construct and execute the SPARQL query
        query = query_template % (doid_values, binding_property)
        results = get_results(endpoint_url, query)
        if results:
            if term_search:
                # Process results for term search
                process_results(results, wikidata_data, batch_data, term_search=term_search)
                continue

            # Process results for HPO codes search
            hpo_code_found.extend(result["hpo_code"]["value"] for result in results)
            process_results(results, wikidata_data)
    # Determine the HPO codes not found in Wikidata
    # This is only relevant for hpo search
    hpo_codes_not_found = list(set(hpo_codes).difference(set(hpo_code_found)))
    return hpo_codes_not_found


# Wikidata dataset

wikidata_data = defaultdict(dict)

# query template
endpoint_url = "https://query.wikidata.org/sparql"

query_template = """

# Select HPO code, labels (terms), and synonyms in English and Spanish

# Start of the query block
SELECT ?item ?hpo_code ?term ?label_en ?label_es (GROUP_CONCAT(DISTINCT ?label_en_alternative; SEPARATOR=", ") AS ?synonyms_en) (GROUP_CONCAT(DISTINCT ?label_es_alternative; SEPARATOR=", ") AS ?synonyms_es)

WHERE
{
  # Filter the items that are subclasses of Q112193867 (Human disease)

  ?item wdt:P31/wdt:P279* wd:Q112193867.

  # Get the English label of the item and filter to keep only English labels
  ?item rdfs:label ?label_en filter (lang(?label_en) = "en").

  # Get the Spanish label of the item and filter to keep only Spanish labels
  ?item rdfs:label ?label_es filter (lang(?label_es) = "es").

  # Get the English synonyms of the item and filter to keep only English synonyms
  ?item skos:altLabel ?label_en_alternative filter (lang(?label_en_alternative) = "en").

  # Get the Spanish synonyms of the item and filter to keep only Spanish synonyms
  ?item skos:altLabel ?label_es_alternative filter (lang(?label_es_alternative) = "es").

  # OPTIONAL block to get the HPO code of the item if available
  OPTIONAL {?item wdt:P3841  ?hpo_code.}

  # Use the VALUES clause to filter the items with specific :
    # HPO codes (HP:0000024 and HP:0000099) OR
    # Term such as "obsolete multicystic dysplastic kidney"@en
    # '%%s' will be replaces but a list of term of hpo codes
  VALUES ?term {  %s }

  # Filter the items to keep only those that match the specified terms
  # '%%s' will be replaced by ?label or wdt:P3841 depending of the type of search performed
  ?item %s ?term
}

# Group the results to get a list of the synonyms en spanish and english instead of having synonym per row
GROUP BY ?item ?hpo_code ?label_en ?label_es ?term


"""


def query_wikidata(filename: str, save_file):
    # Load and parse the XLIFF file
    # Get the root directory
    raw_data = read_json(filename)

    # STEP 1: Search using HPO codes
    hpo_codes = list(raw_data.keys())
    hpo_codes_corrected = [code.replace("_", ":") for code in hpo_codes]
    hpo_codes_not_found = search_wikidata(
        raw_data, hpo_codes_corrected, binding_property="wdt:P3841"
    )

    # STEP 2: Term search for remaining HPO codes not found in STEP 1
    if hpo_codes_not_found:
        search_wikidata(
            raw_data,
            hpo_codes_not_found,
            binding_property="?label",
            batch_size=30,
            term_search=True,
        )

    # save the translation to a new dataset
    save_json_to_file(data=wikidata_data, filepath=save_file)


if __name__ == "__main__":
    query_wikidata("translated_dataset.json", "wikidata.json")
