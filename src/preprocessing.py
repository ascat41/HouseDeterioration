"""Очистка и подготовка данных ГИС ЖКХ для модели аварийности МКД."""

import re

import pandas as pd

TARGET = "condition"
VALID_CONDITIONS = ["Исправный", "Аварийный", "Ветхий"]
PROBLEM_CONDITIONS = ["Аварийный", "Ветхий"]

MIN_YEAR = 1800
MAX_YEAR = 2026

NOT_SPECIFIED = "Не указано"
NO_COORDS = "Координаты не определены"

WALL_RULES = [
    ("Дерево",    r"дерев|брус|бревен|бревн|рублен|сруб|щит|каркасн|засыпн|арболит|арбоблок|фибролит"),
    ("Монолит",   r"монолит"),
    ("Кирпич",    r"кирпич"),
    ("Шлакоблок", r"шлак"),
    ("Блоки",     r"блок|газобет|пенобет|газосилик|ячеист|силикальцит|камн|поротерм"),
    ("Панель",    r"панел|плит|железобет|ж/б"),
    ("Гипсолит",  r"гипс"),
]

PLAN_KIND_RULES = [
    ("Индивидуальный", r"не ?типов|нетипов|не ?серийн|инд|автор|^ип$"),
    ("Серийный",       r"(?<!\d)(?:[iп1]\s*-?\s*)?\d{2,3}(?!\d)"),
    ("Типовой",        r"типов|повторн|экспер"),
]


def load_data(path):
    return pd.read_csv(path, delimiter=";", encoding="utf-8")


def filter_target(df):
    return df[df[TARGET].isin(VALID_CONDITIONS)]


def _valid_year(year):
    return (year > MIN_YEAR) & (year <= MAX_YEAR) & year.notna()


def clean_comissioning_year(df):
    df = df.copy()
    df["comissioning_year"] = df["comissioning_year"].astype("Int64")
    df.loc[~_valid_year(df["comissioning_year"]), "comissioning_year"] = pd.NA
    return df


def clean_building_year(df):
    """Диапазоны вида «2001, 2002» приводятся к первому году, пропуски — к году ввода.

    Требует предварительно очищенного comissioning_year.
    """
    df = df.copy()
    df["building_year"] = df["building_year"].str.split(",").str[0].astype("Int64")
    df["building_year"] = df["building_year"].fillna(df["comissioning_year"])
    df.loc[~_valid_year(df["building_year"]), "building_year"] = pd.NA
    df["missing_building_year"] = df["building_year"].isna()
    return df


def clean_areas(df):
    """Удаляет дома с ошибочной площадью (≤ 1 м² или пропуск) и добавляет отношения."""
    df = df[(df["total_area"] > 1) & (df["residental_premises_area"] > 1)].copy()
    df["residental_ratio"] = df["residental_premises_area"] / df["total_area"]
    df["flats_area_ratio"] = df["flats"] / df["total_area"]
    return df


def map_wall_part(part):
    for group, pattern in WALL_RULES:
        if re.search(pattern, part):
            return group
    return None


def map_walls(value):
    if pd.isna(value) or not str(value).strip():
        return NOT_SPECIFIED
    text = str(value).lower().replace("­", "")
    parts = [p.strip() for p in re.split(r"[,;/+]", text) if p.strip()]
    groups = {g for g in map(map_wall_part, parts) if g}
    if not groups:
        return "Прочее"
    return groups.pop() if len(groups) == 1 else "Смешанные"


def map_plan_kind(raw):
    if pd.isna(raw):
        return NOT_SPECIFIED
    parts = [p.strip() for p in re.split(r"[,;:/]", str(raw).lower()) if p.strip()]
    kinds = set()
    for part in parts:
        for group, rule in PLAN_KIND_RULES:
            if re.search(rule, part):
                kinds.add(group)
                break
    for group in ("Серийный", "Индивидуальный", "Типовой"):   # приоритет при конфликте
        if group in kinds:
            return group
    return NOT_SPECIFIED


def _map_unique(series, func):
    # Уникальных значений намного меньше, чем строк, поэтому маппинг считается по ним
    mapping = {v: func(v) for v in series.dropna().unique()}
    return series.map(mapping).fillna(NOT_SPECIFIED)


def add_walls_group(df):
    df = df.copy()
    df["walls_group"] = _map_unique(df["internal_walls_type"], map_walls)
    return df


def add_plan_kind(df):
    df = df.copy()
    df["plan_kind"] = _map_unique(df["plan_series"], map_plan_kind)
    return df


def clean_municipality_oktmo(df):
    """Переводит ОКТМО в строку и восстанавливает пропуски по почтовому индексу.

    Для каждого индекса берётся самый частый ОКТМО среди домов с этим индексом.
    """
    df = df.copy()
    oktmo = df["municipality_oktmo"].astype("Int64").astype("string")
    postal_to_oktmo = (
        pd.DataFrame({"postal_code": df["postal_code"], "oktmo": oktmo})
        .dropna()
        .value_counts()
        .reset_index()
        .drop_duplicates("postal_code", keep="first")
        .set_index("postal_code")["oktmo"]
    )
    df["municipality_oktmo"] = (
        oktmo.fillna(df["postal_code"].map(postal_to_oktmo)).fillna(NOT_SPECIFIED)
    )
    return df


def clean_categories(df):
    """Приведение типов, заполнение пропусков в категориях и признак реконструкции."""
    df = df.copy()
    df["flats"] = df["flats"].astype("Int64")
    df["house_management_type"] = df["house_management_type"].fillna(NOT_SPECIFIED)
    df["is_reconstructed"] = df["reconstruction_year"].notna().astype(int)
    df["qc_geo"] = df["qc_geo"].fillna(NO_COORDS)
    return df


def binarize_target(df):
    """Аварийный и ветхий — класс 1, исправный — класс 0."""
    df = df.copy()
    df[TARGET] = df[TARGET].isin(PROBLEM_CONDITIONS).astype(int)
    return df


def preprocess(df):
    """Полный пайплайн очистки: сырые данные → датасет для модели."""
    df = filter_target(df)
    df = clean_comissioning_year(df)
    df = clean_building_year(df)
    df = clean_areas(df)
    df = add_walls_group(df)
    df = add_plan_kind(df)
    df = clean_municipality_oktmo(df)
    df = clean_categories(df)
    return binarize_target(df)
