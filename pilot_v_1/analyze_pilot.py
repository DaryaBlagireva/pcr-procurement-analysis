from pathlib import Path
import re
import unicodedata
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


POSITIONS_FILE = BASE_DIR / "output_v_1" / "pilot" / "pilot_contracts_positions.csv"
PILOT_SAMPLE_FILE = BASE_DIR / "output_v_1" / "pilot" / "sampled_contracts_full.csv"

OUTPUT_DIR = BASE_DIR / "output_v_1" / "pilot_analyze"

PCR_RE = re.compile(
    r'пцр'
    r'|пцр[-\s]?(?:рв|рт)'
    r'|от[-\s]?пцр'
    r'|полимеразн\w*[\s-]+цепн\w*[\s-]+реакц\w*'
    r'|(?<![a-z])pcr(?![a-z])'
    r'|rt[-\s]?pcr'
    r'|q[-\s]?pcr'
    r'|rt[-\s]?q[-\s]?pcr'
    r'|real[-\s]?time\s+pcr'
    r'|(?<![a-z])nat(?![a-z])'
    r'|(?<![a-z])naat(?![a-z])'
    r'|nucleic\s+acid\s+test(?:ing)?'
    r'|nucleic\s+acid\s+amplification'
    r'|тестирован\w*[\s-]+нуклеинов\w*[\s-]+кислот\w*'
    r'|исследован\w*[\s-]+нуклеинов\w*[\s-]+кислот\w*'
    r'|амплификац\w*[\s-]+нуклеинов\w*[\s-]+кислот\w*'
    r'|амплификац\w*[\s-]+(?:днк|рнк)'
    r'|(?<![a-z])tma(?![a-z])'
    r'|transcription[-\s]?mediated\s+amplification'
    r'|(?<![a-z])nasba(?![a-z])'
    r'|мультиплекс\w*'
    r'|multiplex'
    r'|mpx',
    flags=re.IGNORECASE,
)

HIV_RE = re.compile(
    r"(?<![а-яa-z])вич(?![а-яa-z])"
    r"|(?<![а-яa-z])hiv(?![а-яa-z])"
    r"|вирус\w*\s+иммунодефицит\w*\s+человек\w*",
    flags=re.IGNORECASE,
)

HBV_RE = re.compile(
    r"(?<![а-яa-z])hbv(?![а-яa-z])"
    r"|(?<![а-яa-z])вгв(?![а-яa-z])",
    flags=re.IGNORECASE,
)

HCV_RE = re.compile(
    r"(?<![а-яa-z])hcv(?![а-яa-z])"
    r"|(?<![а-яa-z])вгс(?![а-яa-z])",
    flags=re.IGNORECASE,
)
HEPATITIS_LIST_RE = re.compile(
    r"(?<!\w)(?:гепатит[а-я]*|hepatitis)"
    r"\s*[-:]?\s*"
    r"(?:вирус[а-я]*\s*)?"
    r"(?:тип[а-я]*\s*)?"
    r"""[«"'(\[]?\s*"""
    r"(?P<first>[abcdeавбсде])(?!\w)"
    r"(?P<rest>"
    r"""(?:[»"')\]]?\s*"""
    r"(?:[,/;+&]|\bи\b|\band\b)"
    r"""\s*[«"'(\[]?\s*"""
    r"[abcdeавбсде](?!\w))*"
    r")",
    flags=re.IGNORECASE,
)

HEPATITIS_LETTER_RE = re.compile(
    r"(?<!\w)[abcdeавбсде](?!\w)",
    flags=re.IGNORECASE,
)

def normalize_text(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    text = re.sub(r"[‐‑‒–—−]", "-", text).replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def find_nosologies(text):
    found = set()
    
    if HIV_RE.search(text):
        found.add("ВИЧ")
    if HBV_RE.search(text):
        found.add("Гепатит B")
    if HCV_RE.search(text):
        found.add("Гепатит C")

    for match in HEPATITIS_LIST_RE.finditer(text):
        letters = [match.group("first")]
        if match.group("rest"):
            letters.extend(HEPATITIS_LETTER_RE.findall(match.group("rest")))

        for letter in letters:
            l_lower = letter.lower()
            if l_lower in {"b", "в", "б"}:
                found.add("Гепатит B")
            elif l_lower in {"c", "с"}:
                found.add("Гепатит C")

    ordered = [n for n in ["ВИЧ", "Гепатит B", "Гепатит C"] if n in found]
    return ordered


def classify_item_name(name_value):
    text = normalize_text(name_value)
    if not text:
        return 0, "Вне целевого сегмента", "", "Пустое название"

    has_pcr = bool(PCR_RE.search(text))
    nosologies = find_nosologies(text)
    has_target = len(nosologies) > 0

    target_in_scope = int(has_pcr and has_target)

    if target_in_scope:
        segment = " + ".join(nosologies)
        reason = f"Найдены ПЦР и целевые нозологии: {segment}"
    elif has_pcr:
        segment = "Вне целевого сегмента"
        reason = "Есть ПЦР, но нет ВИЧ, Гепатита B или C"
    elif has_target:
        segment = "Вне целевого сегмента"
        reason = "Есть целевая нозология, но нет признака ПЦР/амплификации"
    else:
        segment = "Вне целевого сегмента"
        reason = "Нет сочетания ПЦР и целевой нозологии"

    return target_in_scope, segment, "; ".join(nosologies), reason


def parse_numeric(series):
    text = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(text, errors="coerce")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pilot_sample = pd.read_csv(PILOT_SAMPLE_FILE, dtype={"regnum": "string"}, encoding="utf-8-sig")
    positions = pd.read_csv(POSITIONS_FILE, dtype={"regnum": "string"}, encoding="utf-8-sig")

    print("Запуск разметки с полным набором синонимов...")
    classification_results = positions["item_name"].apply(classify_item_name)
    
    positions["target_in_scope"] = [r[0] for r in classification_results]
    positions["target_segment"] = [r[1] for r in classification_results]
    positions["target_nosologies"] = [r[2] for r in classification_results]
    positions["classification_reason"] = [r[3] for r in classification_results]

    positions["unit_price_num"] = parse_numeric(positions["unit_price"])
    positions["quantity_num"] = parse_numeric(positions["quantity"])
    positions["line_amount_num"] = parse_numeric(positions["line_amount_used"])

    calculated_amount = positions["unit_price_num"] * positions["quantity_num"]
    positions["effective_amount"] = positions["line_amount_num"].fillna(calculated_amount)

    positions_summary = (
        positions.groupby("regnum")
        .agg(
            total_positions=("item_name", "count"),
            target_positions=("target_in_scope", "sum"),
            target_amount=(
                "effective_amount",
                lambda s: s[positions.loc[s.index, "target_in_scope"] == 1].sum(),
            ),
            all_positions_amount=("effective_amount", "sum"),
        )
        .reset_index()
    )

    contract_analysis = pilot_sample.merge(positions_summary, on="regnum", how="left")
    
    contract_analysis["target_positions"] = contract_analysis["target_positions"].fillna(0).astype(int)
    contract_analysis["total_positions"] = contract_analysis["total_positions"].fillna(0).astype(int)
    
    contract_analysis["has_target_positions"] = contract_analysis["target_positions"] > 0
    contract_analysis["contract_status"] = np.where(
        contract_analysis["has_target_positions"],
        "Есть целевые позиции",
        "Нецелевой контракт (Ложноположительный)",
    )

    target_only_positions = positions[positions["target_in_scope"] == 1]
    segments_summary = (
        target_only_positions.groupby("target_segment")
        .agg(
            position_count=("item_name", "count"),
            contract_count=("regnum", "nunique"),
            total_segment_amount=("effective_amount", "sum"),
        )
        .reset_index()
    )

    positions.to_csv(OUTPUT_DIR / "pilot_all_positions_analyzed.csv", index=False, encoding="utf-8-sig")
    contract_analysis.to_csv(OUTPUT_DIR / "pilot_contract_analysis.csv", index=False, encoding="utf-8-sig")
    segments_summary.to_csv(OUTPUT_DIR / "pilot_segments_summary.csv", index=False, encoding="utf-8-sig")

    total_pilot_contracts = len(contract_analysis)
    target_contracts = contract_analysis["has_target_positions"].sum()
    non_target_contracts = total_pilot_contracts - target_contracts
    error_rate = (non_target_contracts / total_pilot_contracts) * 100

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ДИАГНОСТИКИ ПИЛОТА (ПОЛНЫЙ ФИЛЬТР)")
    print("=" * 60)
    print(f"Всего контрактов в пилоте: {total_pilot_contracts}")
    print(f"Контрактов с ЦЕЛЕВЫМИ позициями: {target_contracts}")
    print(f"НЕЦЕЛЕВЫХ контрактов в выборке: {non_target_contracts}")
    print(f"Доля ложноположительных контрактов: {error_rate:.1f}%")
    print("=" * 60)
    print("\nРаспределение по целевым сегментам:")
    print(segments_summary.to_string(index=False))


if __name__ == "__main__":
    main()