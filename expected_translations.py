import pandas as pd


def main():
    df = pd.read_table("/Users/zaki/Desktop/Estudios/Master's thesis/Resources/HPO data/hp-es.babelon.tsv",
                       delimiter='\t')
    unnecessary_columns = ['source_language', 'translation_language', 'source_value',
                           'predicate_id', 'translation_status']
    df = df.drop(columns=unnecessary_columns)
    df = df.rename(columns={'subject_id': 'hpo_id', 'translation_value': 'etiqueta'})
    print(df)


if __name__ == "__main__":
    main()
