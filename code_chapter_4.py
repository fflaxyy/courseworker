# -*- coding: utf-8 -*-
from pathlib import Path
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
DATA_FILE = 'tripadvisor_hotel_reviews.csv'
OUTPUT_DIR = Path('figures_chapter4_final_5figures')
TEXT_COLUMN = 'Review'
RATING_COLUMN = 'Rating'
MAX_TFIDF_FEATURES = 5000
SEARCH_QUERIES = ['clean room friendly staff good location', 'dirty bathroom noise bad service', 'beach pool family vacation']
EXTRA_STOP_WORDS_FREQUENCY = {'hotel', 'room', 'stay', 'nt', 'don', 'didn', 'doesn', 'wasn', 'weren', 've', 'll', 're', 'm', 's', 't', 'n'}
EXTRA_STOP_WORDS_TFIDF = {'nt', 'don', 'didn', 'doesn', 'wasn', 'weren', 've', 'll', 're', 'm', 's', 't', 'n'}
STOP_WORDS_FREQUENCY = set(ENGLISH_STOP_WORDS).union(EXTRA_STOP_WORDS_FREQUENCY)
STOP_WORDS_TFIDF = set(ENGLISH_STOP_WORDS).union(EXTRA_STOP_WORDS_TFIDF)

def setup_environment() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

def save_current_figure(filename: str) -> None:
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Сохранено: {path}')

def find_dataset_file() -> Path:
    path = Path(DATA_FILE)
    if path.exists():
        return path
    csv_files = list(Path('.').glob('*.csv'))
    if len(csv_files) == 1:
        print(f'Файл {DATA_FILE} не найден, но найден CSV: {csv_files[0]}')
        print('Использую его.')
        return csv_files[0]
    if len(csv_files) > 1:
        raise FileNotFoundError(f'Файл {DATA_FILE} не найден. В папке несколько CSV: {[str(x) for x in csv_files]}. Укажи нужный файл в DATA_FILE.')
    raise FileNotFoundError(f'CSV-файл не найден. Положи {DATA_FILE} рядом со скриптом.')

def load_dataset() -> pd.DataFrame:
    csv_path = find_dataset_file()
    print(f'Загружаю датасет из файла: {csv_path}')
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in [TEXT_COLUMN, RATING_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f'В датасете не хватает колонок: {missing}. Нужны колонки {TEXT_COLUMN!r} и {RATING_COLUMN!r}.')
    df = df[[TEXT_COLUMN, RATING_COLUMN]].copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    df[RATING_COLUMN] = pd.to_numeric(df[RATING_COLUMN], errors='coerce')
    df = df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN])
    df = df[df[TEXT_COLUMN].str.strip().ne('')]
    df = df.reset_index(drop=True)
    print('Размер датасета:', df.shape)
    return df
IRREGULAR_FORMS = {'was': 'be', 'were': 'be', 'is': 'be', 'are': 'be', 'am': 'be', 'been': 'be', 'has': 'have', 'had': 'have', 'does': 'do', 'did': 'do', 'went': 'go', 'gone': 'go', 'got': 'get', 'better': 'good', 'best': 'good', 'worse': 'bad', 'worst': 'bad', 'rooms': 'room', 'hotels': 'hotel', 'stayed': 'stay', 'staying': 'stay', 'staffs': 'staff', 'services': 'service', 'bathrooms': 'bathroom', 'locations': 'location', 'restaurants': 'restaurant', 'beaches': 'beach', 'pools': 'pool', 'walked': 'walk', 'walking': 'walk', 'cleaned': 'clean', 'cleaning': 'clean'}

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub('<.*?>', ' ', text)
    text = re.sub("[^a-z' ]", ' ', text)
    text = re.sub('\\s+', ' ', text).strip()
    return text

def normalize_word(word: str) -> str:
    if word in IRREGULAR_FORMS:
        return IRREGULAR_FORMS[word]
    if len(word) <= 3:
        return word
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    if word.endswith('es') and len(word) > 4:
        return word[:-2]
    if word.endswith('s') and len(word) > 4:
        return word[:-1]
    return word

def lemmatize_text(text: str) -> str:
    return ' '.join((normalize_word(word) for word in text.split()))

def remove_stopwords_for_frequency(text: str) -> str:
    words = text.split()
    return ' '.join((word for word in words if word not in STOP_WORDS_FREQUENCY and len(word) > 1))

def remove_stopwords_for_tfidf(text: str) -> str:
    words = text.split()
    return ' '.join((word for word in words if word not in STOP_WORDS_TFIDF and len(word) > 1))

def prepare_texts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['clean_text'] = df[TEXT_COLUMN].apply(clean_text)
    df['lemmatized_text'] = df['clean_text'].apply(lemmatize_text)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    df['processed_for_freq'] = df['lemmatized_text'].apply(remove_stopwords_for_frequency)
    df['processed_for_tfidf'] = df['lemmatized_text'].apply(remove_stopwords_for_tfidf)
    return df

def figure_4_1_rating_distribution(df: pd.DataFrame) -> None:
    counts = df[RATING_COLUMN].astype(int).value_counts().sort_index()
    plt.figure(figsize=(9, 5.5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title('Распределение оценок отзывов', fontsize=16)
    plt.xlabel('Оценка', fontsize=12)
    plt.ylabel('Количество отзывов', fontsize=12)
    save_current_figure('figure_4_1_rating_distribution.png')

def figure_4_2_text_length_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5.5))
    plt.hist(df['word_count'], bins=40)
    plt.title('Распределение длины отзывов', fontsize=16)
    plt.xlabel('Количество слов в отзыве', fontsize=12)
    plt.ylabel('Количество отзывов', fontsize=12)
    save_current_figure('figure_4_2_text_length_distribution.png')

def get_word_counts(texts: pd.Series) -> Counter:
    all_words = ' '.join(texts.dropna()).split()
    return Counter(all_words)

def figure_4_5_top_words_after_stopwords(counter: Counter) -> None:
    top_words = counter.most_common(15)
    words = [word for word, _ in top_words][::-1]
    counts = [count for _, count in top_words][::-1]
    plt.figure(figsize=(9, 5.5))
    plt.barh(words, counts)
    plt.title('Наиболее частотные слова после удаления стоп-слов', fontsize=16)
    plt.xlabel('Частота', fontsize=12)
    plt.ylabel('')
    save_current_figure('figure_4_5_top_words_after_stopwords.png')

def build_tfidf(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, ngram_range=(1, 1), min_df=2, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(df['processed_for_tfidf'])
    feature_names = vectorizer.get_feature_names_out()
    print(f'Размер словаря TF-IDF: {len(feature_names)}')
    print(f'Размер TF-IDF матрицы: {tfidf_matrix.shape}')
    return (vectorizer, tfidf_matrix, feature_names)

def get_top_tfidf_terms(tfidf_matrix, feature_names, top_n: int=15) -> pd.DataFrame:
    mean_values = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    top_indices = mean_values.argsort()[::-1][:top_n]
    return pd.DataFrame({'term': feature_names[top_indices], 'mean_tfidf': mean_values[top_indices]})

def figure_4_8_top_tfidf_terms(top_terms: pd.DataFrame) -> None:
    plot_df = top_terms.iloc[::-1].copy()
    plt.figure(figsize=(9, 6))
    plt.barh(plot_df['term'], plot_df['mean_tfidf'])
    plt.title('Ключевые термины по среднему TF-IDF', fontsize=13)
    plt.xlabel('Среднее значение TF-IDF', fontsize=11)
    plt.ylabel('')
    save_current_figure('figure_4_8_top_tfidf_terms.png')

def make_sentiment_label(rating: float) -> str:
    if rating <= 2:
        return 'negative'
    if rating == 3:
        return 'neutral'
    return 'positive'

def prepare_query(query: str) -> str:
    query = clean_text(query)
    query = lemmatize_text(query)
    query = remove_stopwords_for_tfidf(query)
    return query

def search_texts(query: str, vectorizer, tfidf_matrix, df: pd.DataFrame, top_n: int=3) -> pd.DataFrame:
    prepared_query = prepare_query(query)
    query_vector = vectorizer.transform([prepared_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    rows = []
    for number, idx in enumerate(top_indices, 1):
        rows.append({'query': query, 'number': number, 'similarity': float(similarities[idx]), 'rating': int(df.loc[idx, RATING_COLUMN]), 'class': make_sentiment_label(df.loc[idx, RATING_COLUMN]), 'review': str(df.loc[idx, TEXT_COLUMN])})
    return pd.DataFrame(rows)

def shorten_query(query: str, max_len: int=19) -> str:
    if len(query) <= max_len:
        return query
    return query[:max_len].rstrip() + '...'

def figure_4_9_search_results(search_df: pd.DataFrame) -> None:
    labels = [f"{shorten_query(row['query'])} #{int(row['number'])}" for _, row in search_df.iterrows()]
    values = search_df['similarity'].values
    plt.figure(figsize=(9, 6))
    plt.barh(labels, values)
    plt.title('Результаты информационного поиска по 3 запросам', fontsize=13)
    plt.xlabel('Косинусное сходство', fontsize=11)
    plt.ylabel('')
    plt.xlim(0, max(0.1, float(values.max()) * 1.08))
    plt.gca().invert_yaxis()
    save_current_figure('figure_4_9_search_results.png')

def main() -> None:
    setup_environment()
    df = load_dataset()
    df = prepare_texts(df)
    figure_4_1_rating_distribution(df)
    figure_4_2_text_length_distribution(df)
    freq_counter = get_word_counts(df['processed_for_freq'])
    figure_4_5_top_words_after_stopwords(freq_counter)
    vectorizer, tfidf_matrix, feature_names = build_tfidf(df)
    top_terms = get_top_tfidf_terms(tfidf_matrix, feature_names, top_n=15)
    print('\nТермины с наибольшим средним значением TF-IDF:')
    print(top_terms)
    figure_4_8_top_tfidf_terms(top_terms)
    search_parts = []
    for query in SEARCH_QUERIES:
        search_parts.append(search_texts(query, vectorizer, tfidf_matrix, df, top_n=3))
    search_df = pd.concat(search_parts, ignore_index=True)
    print('\nРезультаты информационного поиска:')
    print(search_df[['query', 'number', 'similarity', 'rating', 'class']])
    figure_4_9_search_results(search_df)
    print('\nГотово!')
    print('Созданы только нужные графики, без таблиц.')
    print('Папка с результатами:', OUTPUT_DIR.resolve())
    print('Файлы:')
    print(' - figure_4_1_rating_distribution.png')
    print(' - figure_4_2_text_length_distribution.png')
    print(' - figure_4_5_top_words_after_stopwords.png')
    print(' - figure_4_8_top_tfidf_terms.png')
    print(' - figure_4_9_search_results.png')
if __name__ == '__main__':
    main()
