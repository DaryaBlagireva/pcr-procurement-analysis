from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / 'output_v_1' / 'relevant_contracts_all.csv'

OUTPUT_DIR = BASE_DIR / 'output_v_1' / 'pilot'

SEED = 42
PILOT_FRACTION = 0.10
MIN_PER_GROUP = 2
MAX_PER_GROUP = 15

GROUP_COLUMNS = [
    "year",
    "sampling_stratum",
    "fed_district",
]


def main():
    # Читаем номера строками, чтобы сохранить все 19 цифр.
    contracts = pd.read_csv(
        INPUT_FILE,
        dtype={"regnum": "string"},
        encoding="utf-8-sig",
    )

    required = ["regnum"] + GROUP_COLUMNS
    missing = set(required) - set(contracts.columns)

    if missing:
        raise ValueError(
            f"В реестре отсутствуют столбцы: {sorted(missing)}"
        )

    if contracts[required].isna().any().any():
        raise ValueError("В обязательных полях есть пропуски.")

    contracts["regnum"] = contracts["regnum"].str.strip()

    if not contracts["regnum"].str.fullmatch(r"[0-9]{19}").all():
        raise ValueError("Номера контрактов должны содержать 19 цифр.")

    if contracts["regnum"].duplicated().any():
        raise ValueError("В реестре есть повторные номера контрактов.")

    # Только случайная часть пилота: контракты до 5 млн включительно.
    sampling_base = contracts[
        contracts["sampling_stratum"].isin(
            ["lt_1m", "1m_to_5m"]
        )
    ].copy()

    if sampling_base.empty:
        raise ValueError("Нет контрактов для формирования пилота.")

    # Один генератор перед циклом — не создавать заново внутри групп.
    rng = np.random.default_rng(SEED)

    selected_parts = []
    plan = []

    for group_key, group in sampling_base.groupby(
        GROUP_COLUMNS,
        sort=True,
        observed=True,
    ):
        # Порядок нужен для воспроизведения прежней выборки.
        group = group.sort_values("regnum").reset_index(drop=True)

        population_size = len(group)

        sample_size = min(
            population_size,
            MAX_PER_GROUP,
            max(
                MIN_PER_GROUP,
                int(np.ceil(PILOT_FRACTION * population_size)),
            ),
        )

        selected_indices = rng.choice(
            population_size,
            size=sample_size,
            replace=False,
        )

        selected_parts.append(
            group.iloc[selected_indices]
        )

        plan.append({
            "year": group_key[0],
            "sampling_stratum": group_key[1],
            "fed_district": group_key[2],
            "population_size": population_size,
            "sample_size": sample_size,
        })

    # Только исходные столбцы реестра.
    pilot = (
        pd.concat(selected_parts, ignore_index=True)
        .sort_values("regnum")
        .reset_index(drop=True)
    )

    numbers = (
        pilot[["regnum"]]
        .rename(columns={"regnum": "contract_number"})
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    numbers.to_csv(
        OUTPUT_DIR / "selected_contract_numbers.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pilot.to_csv(
        OUTPUT_DIR / "sampled_contracts_full.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        pd.DataFrame(plan)
        .groupby(
            ["year", "sampling_stratum"],
            as_index=False,
        )[["population_size", "sample_size"]]
        .sum()
    )

    print("\nПЛАН ПИЛОТНОЙ ВЫБОРКИ")
    print(summary.to_string(index=False))

    print(f"\nВыбрано контрактов: {len(pilot)}")
    print(f"\nРезультаты сохранены в: {OUTPUT_DIR}")
    print("1. selected_contract_numbers.csv — только номера")
    print("2. sampled_contracts_full.csv — данные выбранных контрактов")


if __name__ == "__main__":
    main()
