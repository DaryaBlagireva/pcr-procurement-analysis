import os
from pathlib import Path 
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent 

RAW_DATA_DIR = BASE_DIR / 'raw_data'


OUTPUT_DIR = Path(__file__).resolve().parent / 'output_v_1'

TEXT_COLUMNS = [
    'subject',
    'industry',
    'subindustry',
    'product_names',
    'product_codes'
]

PCR_PATTERN = (
    r'пцр'
    r'|пцр[-\s]?(?:рв|рт)'
    r'|от[-\s]?пцр'
    r'|полимеразн\w*\s+цепн\w*\s+реакц\w*'
    r'|(?<![a-z])pcr(?![a-z])'
    r'|rt[-\s]?pcr'
    r'|q[-\s]?pcr'
    r'|rt[-\s]?q[-\s]?pcr'
    r'|real[-\s]?time\s+pcr'
    r'|(?<![a-z])nat(?![a-z])'
    r'|(?<![a-z])naat(?![a-z])'
    r'|nucleic\s+acid\s+test(?:ing)?'
    r'|nucleic\s+acid\s+amplification'
    r'|тестирован\w*\s+нуклеинов\w*\s+кислот\w*'
    r'|исследован\w*\s+нуклеинов\w*\s+кислот\w*'
    r'|амплификац\w*\s+нуклеинов\w*\s+кислот\w*'
    r'|амплификац\w*\s+(?:днк|рнк)'
    r'|(?<![a-z])tma(?![a-z])'
    r'|transcription[-\s]?mediated\s+amplification'
    r'|(?<![a-z])nasba(?![a-z])'
    r'|мультиплекс\w*'
    r'|multiplex'
    r'|mpx'
)

TARGET_PATTERN = (
    r'вич'
    r'|hiv'
    r'|вирус\w*\s+иммунодефицит\w*\s+человек\w*'
    r'|гепатит\w*\s*[bв]'
    r'|гепатит\w*\s*[cс]'
    r'|(?<![a-zа-я])hbv(?![a-zа-я])'
    r'|(?<![a-zа-я])hcv(?![a-zа-я])'
    r'|вгв'
    r'|вгс'
)

FED_DISTRICT_MAP = {
    1: 'Южный ФО', 2: 'Приволжский ФО', 3: 'Сибирский ФО', 4: 'Сибирский ФО',
    5: 'Северо-Кавказский ФО', 6: 'Северо-Кавказский ФО', 7: 'Северо-Кавказский ФО',
    8: 'Южный ФО', 9: 'Северо-Кавказский ФО', 10: 'Северо-Западный ФО',
    11: 'Северо-Западный ФО', 12: 'Приволжский ФО', 13: 'Приволжский ФО',
    14: 'Дальневосточный ФО', 15: 'Северо-Кавказский ФО', 16: 'Приволжский ФО',
    17: 'Сибирский ФО', 18: 'Приволжский ФО', 19: 'Сибирский ФО',
    20: 'Северо-Кавказский ФО', 21: 'Приволжский ФО', 22: 'Сибирский ФО',
    23: 'Южный ФО', 24: 'Сибирский ФО', 25: 'Дальневосточный ФО',
    26: 'Северо-Кавказский ФО', 27: 'Дальневосточный ФО', 28: 'Дальневосточный ФО',
    29: 'Северо-Западный ФО', 30: 'Южный ФО', 31: 'Центральный ФО',
    32: 'Центральный ФО', 33: 'Центральный ФО', 34: 'Южный ФО',
    35: 'Северо-Западный ФО', 36: 'Центральный ФО', 37: 'Центральный ФО',
    38: 'Сибирский ФО', 39: 'Северо-Западный ФО', 40: 'Центральный ФО',
    41: 'Дальневосточный ФО', 42: 'Сибирский ФО', 43: 'Приволжский ФО',
    44: 'Центральный ФО', 45: 'Уральский ФО', 46: 'Центральный ФО',
    47: 'Северо-Западный ФО', 48: 'Центральный ФО', 49: 'Дальневосточный ФО',
    50: 'Центральный ФО', 51: 'Северо-Западный ФО', 52: 'Приволжский ФО',
    53: 'Северо-Западный ФО', 54: 'Сибирский ФО', 55: 'Сибирский ФО',
    56: 'Приволжский ФО', 57: 'Центральный ФО', 58: 'Приволжский ФО',
    59: 'Приволжский ФО', 60: 'Северо-Западный ФО', 61: 'Южный ФО',
    62: 'Центральный ФО', 63: 'Приволжский ФО', 64: 'Приволжский ФО',
    65: 'Дальневосточный ФО', 66: 'Уральский ФО', 67: 'Центральный ФО',
    68: 'Центральный ФО', 69: 'Центральный ФО', 70: 'Сибирский ФО',
    71: 'Центральный ФО', 72: 'Уральский ФО', 73: 'Приволжский ФО',
    74: 'Уральский ФО', 75: 'Дальневосточный ФО', 76: 'Центральный ФО',
    77: 'Центральный ФО', 78: 'Северо-Западный ФО', 79: 'Дальневосточный ФО',
    80: 'Дальневосточный ФО', 81: 'Приволжский ФО', 82: 'Южный ФО',
    83: 'Северо-Западный ФО', 84: 'Сибирский ФО', 85: 'Сибирский ФО',
    86: 'Уральский ФО', 87: 'Дальневосточный ФО', 88: 'Сибирский ФО',
    89: 'Уральский ФО', 90: 'Центральный ФО', 91: 'Северо-Западный ФО',
    92: 'Южный ФО', 93: 'Южный ФО', 94: 'Дальневосточный ФО',
    95: 'Северо-Кавказский ФО', 97: 'Центральный ФО', 98: 'Северо-Западный ФО',
    99: 'Центральный ФО'
}

SAMPLED_STRATA = ['lt_1m', '1m_to_5m']
CERTAINTY_STRATUM = 'gt_5m'


def normalize_region_code(x):
    if pd.isna(x):
        return np.nan

    value = pd.to_numeric(str(x).strip(), errors='coerce')

    if pd.isna(value) or not np.isfinite(value):
        return np.nan

    if value < 0 or value != int(value):
        return np.nan

    return int(value)


def assign_sampling_stratum(amount):
    if pd.isna(amount):
        return np.nan

    if amount < 1_000_000:
        return 'lt_1m'

    if amount <= 5_000_000:
        return '1m_to_5m'

    return 'gt_5m'


def load_and_prepare_data():
    read_dtypes = {
        'regnum': 'string',
        'region_code': 'string'
    }

    # ИСПРАВЛЕНО: используем RAW_DATA_DIR для чтения файлов
    hep_b = pd.read_csv(
        RAW_DATA_DIR / 'hep_b.txt',
        dtype=read_dtypes
    )
    hep_c = pd.read_csv(
        RAW_DATA_DIR / 'hep_c.txt',
        dtype=read_dtypes
    )
    hiv = pd.read_csv(
        RAW_DATA_DIR / 'hiv.txt',
        dtype=read_dtypes
    )

    df = pd.concat([hep_b, hep_c, hiv], ignore_index=True)

    required_columns = set(
        TEXT_COLUMNS
        + ['regnum', 'sign_date', 'amount_rur', 'region_code']
    )

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            'Отсутствуют обязательные столбцы: '
            f'{sorted(missing_columns)}'
        )

    if df['regnum'].isna().any():
        raise ValueError(
            'В исходных данных есть контракты без regnum.'
        )

    df = df.drop_duplicates(subset='regnum').copy()

    df['search_text'] = (
        df[TEXT_COLUMNS]
        .fillna('')
        .astype(str)
        .agg(' '.join, axis=1)
        .str.lower()
    )

    relevant_contracts = df[
        df['search_text'].str.contains(
            PCR_PATTERN,
            regex=True,
            na=False
        )
        & df['search_text'].str.contains(
            TARGET_PATTERN,
            regex=True,
            na=False
        )
    ].copy()

    relevant_contracts['sign_date'] = pd.to_datetime(
        relevant_contracts['sign_date'],
        errors='coerce'
    )

    relevant_contracts['year'] = (
        relevant_contracts['sign_date']
        .dt.year
        .astype('Int64')
    )

    relevant_contracts['amount_rur'] = pd.to_numeric(
        relevant_contracts['amount_rur'],
        errors='coerce'
    )

    invalid_rows = (
        relevant_contracts['year'].isna()
        | relevant_contracts['amount_rur'].isna()
        | ~np.isfinite(relevant_contracts['amount_rur'])
        | relevant_contracts['amount_rur'].lt(0)
    )

    if invalid_rows.any():
        invalid_ids = (
            relevant_contracts.loc[invalid_rows, 'regnum']
            .head(20)
            .tolist()
        )

        raise ValueError(
            'Есть контракты с некорректной датой или суммой. '
            f'Проверьте regnum: {invalid_ids}'
        )

    relevant_contracts['region_code_num'] = (
        relevant_contracts['region_code']
        .apply(normalize_region_code)
        .astype('Int64')
    )

    relevant_contracts['fed_district'] = (
        relevant_contracts['region_code_num']
        .map(FED_DISTRICT_MAP)
        .fillna('Не определен')
    )

    relevant_contracts['sampling_stratum'] = (
        relevant_contracts['amount_rur']
        .apply(assign_sampling_stratum)
    )

    return relevant_contracts.reset_index(drop=True)


def build_district_population(relevant_contracts):
    district_population = (
        relevant_contracts
        .groupby(
            ['year', 'sampling_stratum', 'fed_district'],
            as_index=False
        )
        .agg(
            district_population=('regnum', 'size'),
            total_contract_amount=('amount_rur', 'sum')
        )
        .sort_values(
            ['year', 'sampling_stratum', 'fed_district']
        )
        .reset_index(drop=True)
    )

    district_population['stratum_population'] = (
        district_population
        .groupby(['year', 'sampling_stratum'])[
            'district_population'
        ]
        .transform('sum')
    )

    district_population['district_share'] = (
        district_population['district_population']
        / district_population['stratum_population']
    )

    district_population['is_certainty_stratum'] = (
        district_population['sampling_stratum']
        .eq(CERTAINTY_STRATUM)
    )

    return district_population


def save_preparation_outputs(
    relevant_contracts,
    output_dir=OUTPUT_DIR  # ИСПРАВЛЕНО: используем константу OUTPUT_DIR по умолчанию
):
    # Создаем директорию, если её нет
    output_dir.mkdir(parents=True, exist_ok=True)

    # ИСПРАВЛЕНО: используем современный путь через / (pathlib)
    relevant_contracts.to_csv(
        output_dir / 'relevant_contracts_all.csv',
        index=False,
        encoding='utf-8-sig'
    )

    district_population = build_district_population(
        relevant_contracts
    )

    district_population.to_csv(
        output_dir / 'district_population_by_stratum.csv',
        index=False,
        encoding='utf-8-sig'
    )

    district_population_for_sampling = district_population[
        district_population['sampling_stratum']
        .isin(SAMPLED_STRATA)
    ].copy()

    district_population_for_sampling.to_csv(
        output_dir / 'district_population_for_sampling.csv',
        index=False,
        encoding='utf-8-sig'
    )

    certainty_contracts = relevant_contracts[
        relevant_contracts['sampling_stratum']
        .eq(CERTAINTY_STRATUM)
    ].copy()

    certainty_contracts.to_csv(
        output_dir / 'certainty_contracts_gt_5m.csv',
        index=False,
        encoding='utf-8-sig'
    )

    return district_population

if __name__ == '__main__':
    relevant_contracts = load_and_prepare_data()

    district_population = save_preparation_outputs(
        relevant_contracts
    )

    district_population_for_sampling = district_population[
        district_population['sampling_stratum']
        .isin(SAMPLED_STRATA)
    ].copy()