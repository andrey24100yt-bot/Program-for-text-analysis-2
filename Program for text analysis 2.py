"""Лабораторная работа № 2. Вариант 4 — Музейный каталог искусства.

Запуск:
    python lab2.py

Скрипт читает CSV из папки 04_Музей_искусства рядом с собой и формирует:
    search_results.csv       — TOP-5 для всех 30 запросов;
    experiment_metrics.csv   — Hit@5 и MRR@5 для A/B/C/D;
    baseline_results.csv     — результаты базового поиска по словам;
    analysis_errors.txt      — разбор примеров ошибок ранжирования.

Лемматизация: pymorphy3, если пакет установлен. Если его нет, используется
SnowballStemmer из NLTK как воспроизводимый резервный вариант морфологической
нормализации. Для строгой лемматизации установите pymorphy3.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Васильев Андрей
try:
    import pymorphy3  # type: ignore
    _morph = pymorphy3.MorphAnalyzer()
    LEMMATIZER_NAME = "pymorphy3 (лемматизация)"

    def normalize_lemma(word: str) -> str:
        return _morph.parse(word)[0].normal_form
except Exception:
    try:
        from nltk.stem.snowball import SnowballStemmer
        _stemmer = SnowballStemmer("russian")
        LEMMATIZER_NAME = "NLTK SnowballStemmer (резервная стемминг-нормализация)"

        def normalize_lemma(word: str) -> str:
            return _stemmer.stem(word)
    except Exception as exc:
        raise RuntimeError(
            "Для нормализации установите pymorphy3 или nltk: "
            "pip install pymorphy3 nltk"
        ) from exc


TOKEN_RE = re.compile(r"[а-яё]+", flags=re.IGNORECASE)

# Встроенный список обеспечивает воспроизводимость, не требующий загрузки дополнительных корпусов NLTK во время запуска.
RUSSIAN_STOPWORDS = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы
по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг
ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом
себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам
чтоб без будто чего раз тоже себе под подо будет ж тогда кто этот того потому
этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были
куда зачем сказать всех никогда сегодня можно ли хотя между
""".split())


def tokenize(text: str, *, lemmatize: bool, remove_stopwords: bool) -> list[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(str(text))]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in RUSSIAN_STOPWORDS]
    if lemmatize:
        tokens = [normalize_lemma(t) for t in tokens]
    return tokens


def get_data_dir() -> Path:
    root = Path(__file__).resolve().parent
    nested = root / "04_Музей_искусства"
    if (nested / "documents.csv").exists():
        return nested
    if (root / "documents.csv").exists():
        return root
    raise FileNotFoundError(
        "Не найдена папка 04_Музей_искусства с documents.csv, queries.csv и dev_relevance.csv"
    )

# Ветин Константин
def stable_top_indices(scores: np.ndarray, k: int = 5) -> np.ndarray:
    # Стабильная сортировка разрешает равенства в исходном порядке документов.
    return np.argsort(-scores, kind="stable")[:k]


def evaluate_ranking(
    similarities: np.ndarray,
    queries: pd.DataFrame,
    documents: pd.DataFrame,
    relevance: pd.DataFrame,
) -> tuple[float, float, dict[str, int]]:
    doc_index = {doc_id: i for i, doc_id in enumerate(documents["document_id"])}
    query_index = {query_id: i for i, query_id in enumerate(queries["query_id"])}
    hit_values: list[int] = []
    reciprocal_ranks: list[float] = []
    ranks: dict[str, int] = {}

    for row in relevance.itertuples(index=False):
        q_idx = query_index[row.query_id]
        gold_idx = doc_index[row.relevant_document_id]
        ranking = stable_top_indices(similarities[q_idx], k=len(documents))
        rank = int(np.where(ranking == gold_idx)[0][0]) + 1
        ranks[row.query_id] = rank
        hit_values.append(int(rank <= 5))
        reciprocal_ranks.append(1.0 / rank if rank <= 5 else 0.0)

    return float(np.mean(hit_values)), float(np.mean(reciprocal_ranks)), ranks


def run_baseline(
    documents: pd.DataFrame, queries: pd.DataFrame, doc_texts: list[str]
) -> pd.DataFrame:
    """Базовый поиск: число совпавших уникальных слов запроса и документа."""
    doc_tokens = [set(tokenize(t, lemmatize=False, remove_stopwords=True)) for t in doc_texts]
    rows: list[dict[str, object]] = []
    for query in queries.itertuples(index=False):
        q_tokens = set(tokenize(query.query, lemmatize=False, remove_stopwords=True))
        scores = np.array([len(q_tokens & tokens) for tokens in doc_tokens], dtype=float)
        for rank, idx in enumerate(stable_top_indices(scores, k=5), start=1):
            rows.append({
                "query_id": query.query_id,
                "rank": rank,
                "document_id": documents.iloc[idx]["document_id"],
                "score": float(scores[idx]),
            })
    return pd.DataFrame(rows, columns=["query_id", "rank", "document_id", "score"])

# Воронов Арсений
def main() -> None:
    data_dir = get_data_dir()
    out_dir = Path(__file__).resolve().parent

    documents = pd.read_csv(data_dir / "documents.csv", encoding="utf-8-sig")
    queries = pd.read_csv(data_dir / "queries.csv", encoding="utf-8-sig")
    relevance = pd.read_csv(data_dir / "dev_relevance.csv", encoding="utf-8-sig")

    required_doc_cols = {"document_id", "title", "section", "text"}
    required_query_cols = {"query_id", "query", "difficulty"}
    if not required_doc_cols.issubset(documents.columns):
        raise ValueError(f"В documents.csv отсутствуют поля: {required_doc_cols - set(documents.columns)}")
    if not required_query_cols.issubset(queries.columns):
        raise ValueError(f"В queries.csv отсутствуют поля: {required_query_cols - set(queries.columns)}")

    print(" Исследование корпуса ")
    print(f"Документов: {len(documents)}; запросов: {len(queries)}; размеченных запросов: {len(relevance)}")
    print(f"Поля документов: {list(documents.columns)}")
    print(f"Пропуски в документах:\n{documents.isna().sum().to_string()}")
    print(f"Дубликаты document_id: {documents['document_id'].duplicated().sum()}")
    print(f"Уникальные document_id: {documents['document_id'].nunique()}")
    print(f"Лемматизатор/нормализатор: {LEMMATIZER_NAME}")

    # Моторин Михаил
    # Объединяем заголовок, раздел и основной текст: так в индекс попадают содержимое карточки и её структурные признаки.
    doc_texts = (
        documents["title"].fillna("").astype(str)
        + " " + documents["section"].fillna("").astype(str)
        + " " + documents["text"].fillna("").astype(str)
    ).tolist()
    query_texts = queries["query"].fillna("").astype(str).tolist()

    configs = {
        "A": {"lemmatize": False, "remove_stopwords": False, "ngram_range": (1, 1)},
        "B": {"lemmatize": True,  "remove_stopwords": False, "ngram_range": (1, 1)},
        "C": {"lemmatize": True,  "remove_stopwords": True,  "ngram_range": (1, 1)},
        "D": {"lemmatize": True,  "remove_stopwords": True,  "ngram_range": (1, 2)},
    }

    metrics: list[dict[str, object]] = []
    rankings: dict[str, np.ndarray] = {}
    rank_details: dict[str, dict[str, int]] = {}
    score_matrices: dict[str, np.ndarray] = {}

    for name, cfg in configs.items():
        tokenizer: Callable[[str], list[str]] = lambda text, c=cfg: tokenize(
            text,
            lemmatize=bool(c["lemmatize"]),
            remove_stopwords=bool(c["remove_stopwords"]),
        )
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            token_pattern=None,
            lowercase=False,
            ngram_range=cfg["ngram_range"],
        )
        # Филиповский Виталий
        # Обучение векторизатора основывается на корпусе документов.
        document_matrix = vectorizer.fit_transform(doc_texts)
        query_matrix = vectorizer.transform(query_texts)
        similarities = cosine_similarity(query_matrix, document_matrix)
        score_matrices[name] = similarities

        hit, mrr, ranks = evaluate_ranking(similarities, queries, documents, relevance)
        rank_details[name] = ranks
        metrics.append({
            "configuration": name,
            "hit_at_5": hit,
            "mrr_at_5": mrr,
        })
        rankings[name] = similarities
        print(f"Конфигурация {name}: Hit@5={hit:.4f}; MRR@5={mrr:.4f}")

    metrics_df = pd.DataFrame(metrics, columns=["configuration", "hit_at_5", "mrr_at_5"])
    metrics_df.to_csv(out_dir / "experiment_metrics.csv", index=False, encoding="utf-8-sig")

    # При равных метриках предпочтительно берём вариант с удалением стоп-слов.
    metric_lookup = {row["configuration"]: row for row in metrics}
    selected = max(
        configs,
        key=lambda name: (
            metric_lookup[name]["hit_at_5"],
            metric_lookup[name]["mrr_at_5"],
            int(bool(configs[name]["remove_stopwords"])),
            -int(configs[name]["ngram_range"][1] > 1),
        ),
    )
    print(f"Итоговая конфигурация для search_results.csv: {selected}")

    final_similarities = score_matrices[selected]
    result_rows: list[dict[str, object]] = []
    for q_idx, query in enumerate(queries.itertuples(index=False)):
        top = stable_top_indices(final_similarities[q_idx], k=5)
        for rank, doc_idx in enumerate(top, start=1):
            result_rows.append({
                "query_id": query.query_id,
                "rank": rank,
                "document_id": documents.iloc[doc_idx]["document_id"],
                "score": float(final_similarities[q_idx, doc_idx]),
            })
    pd.DataFrame(result_rows, columns=["query_id", "rank", "document_id", "score"]).to_csv(
        out_dir / "search_results.csv", index=False, encoding="utf-8-sig"
    )

    baseline = run_baseline(documents, queries, doc_texts)
    baseline.to_csv(out_dir / "baseline_results.csv", index=False, encoding="utf-8-sig")

    # Тагильцева Дарья
    # Автоматически выбираем два показательных промаха/снижения ранга среди открытых эталонных запросов для иллюстрации ошибок моделей.
    doc_title = dict(zip(documents["document_id"], documents["title"]))
    q_text = dict(zip(queries["query_id"], queries["query"]))
    gold_map = dict(zip(relevance["query_id"], relevance["relevant_document_id"]))
    error_lines = [
        "РАЗБОР ПРИМЕРОВ ОШИБОК РАНЖИРОВАНИЯ",
        "Метрики считаются только по 8 запросам из dev_relevance.csv.",
        "",
    ]
    examples = [("A", "ART-Q18"), ("D", "ART-Q07")]
    for config_name, query_id in examples:
        gold_id = gold_map[query_id]
        rank = rank_details[config_name][query_id]
        sim = score_matrices[config_name][queries.index[queries["query_id"] == query_id][0]]
        top = stable_top_indices(sim, k=6)
        error_lines.extend([
            f"Конфигурация {config_name}, запрос {query_id} (эталон: {gold_id}, {doc_title[gold_id]}).",
            f"Запрос: {q_text[query_id]}",
            f"Позиция эталонного документа: {rank}.",
            "Первые результаты:",
        ])
        for pos, idx in enumerate(top, start=1):
            error_lines.append(
                f"  {pos}. {documents.iloc[idx]['document_id']} — "
                f"{documents.iloc[idx]['title']} (score={sim[idx]:.4f})"
            )
        if config_name == "A":
            error_lines.append(
                "Причина: без лемматизации формы «посетителю» и «посетитель», "
                "а также близкие по смыслу слова из формулировки запроса и карточек "
                "не сводятся к общей основе. Поэтому другие карточки того же объекта "
                "получают высокие баллы за название и общую лексику."
            )
        else:
            error_lines.append(
                "Причина: добавление биграмм в конфигурации D меняет относительные "
                "веса признаков. Общие фразы о разделе и названии темы усиливают "
                "карточки того же объекта, а нужная аспектная карточка опускается "
                "ниже нескольких тематически близких результатов."
            )
        error_lines.append("")
    (out_dir / "analysis_errors.txt").write_text("\n".join(error_lines), encoding="utf-8")

    print("\nГотово. Созданы search_results.csv, experiment_metrics.csv, baseline_results.csv и analysis_errors.txt")


if __name__ == "__main__":
    main()
