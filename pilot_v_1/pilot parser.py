import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

BASE_DIR = Path(__file__).resolve().parent

WORD_FILE = BASE_DIR / "pilot_raw_data" / "pilot.docx"

REGISTRY_FILE = BASE_DIR / "output_v_1" / "relevant_contracts_all.csv"

OUTPUT_FILE = BASE_DIR / "output_v_1" / "pilot" / "pilot_contracts_positions.csv"


def number(value):
    value = re.sub(r"\s+", "", value).replace(",", ".")

    if value.lower() in {"", "-", "—", "–", "нет", "н/д"}:
        return None

    result = float(value)

    if not 0 <= result < float("inf"):
        raise ValueError("Некорректное число")

    return result


def main():
    rows, contracts, issues = [], set(), {}
    regnum = None

    def issue(contract, message):
        issues.setdefault(contract or "Без номера", set()).add(message)

    document = Document(WORD_FILE)

    for element in document.element.body:
        if element.tag == qn("w:p"):
            text = Paragraph(element, document).text
            ids = re.findall(r"(?<!\d)\d{19}(?!\d)", text)

            if len(ids) == 1:
                regnum = ids[0]
                contracts.add(regnum)

            elif len(ids) > 1:
                regnum = None
                issue(None, "несколько номеров в одном абзаце")

        elif element.tag == qn("w:tbl"):
            for row in Table(element, document).rows:
                cells = [
                    re.sub(r"\s+", " ", cell.text).strip()
                    for cell in row.cells
                ]

                if not any(cells):
                    continue

                text = " ".join(cells).lower()

                if "наименование" in text and (
                    "цена" in text or "окпд" in text
                ):
                    continue

                if regnum is None:
                    issue(None, "таблица без номера контракта")
                    continue

                if len(cells) == 7 and re.fullmatch(
                    r"\d+[.)]?", cells[0]
                ):
                    cells = cells[1:]

                if len(cells) != 6:
                    issue(regnum, "неверная структура строки")
                    continue

                name, code, unit, price, quantity, amount = cells

                try:
                    price, quantity, amount = map(
                        number, [price, quantity, amount]
                    )
                except ValueError:
                    issue(regnum, f"не распознаны числа: {name}")
                    price = quantity = amount = None

                product = (
                    price * quantity
                    if price is not None and quantity is not None
                    else None
                )

                if amount == 0 and quantity is None:
                    amount = None
                elif amount is None:
                    amount = product

                if (
                    amount is not None
                    and product is not None
                    and abs(amount - product)
                    > max(0.02, abs(amount) * 1e-6)
                ):
                    issue(
                        regnum,
                        f"цена × количество ≠ сумма: {name}"
                    )
                    amount = None

                if amount is None:
                    issue(regnum, "есть позиции без достоверной суммы")

                rows.append([
                    regnum, name, code, unit,
                    price, quantity, amount
                ])

    positions = pd.DataFrame(rows, columns=[
        "regnum", "item_name", "okpd2_raw", "unit",
        "unit_price", "quantity", "line_amount_used"
    ])

    duplicate_columns = [
        "item_name", "okpd2_raw", "unit",
        "unit_price", "quantity", "line_amount_used"
    ]

    for contract, group in positions.groupby("regnum"):
        duplicate_mask = group.duplicated(
            duplicate_columns,
            keep=False
        )

        if duplicate_mask.any():
            repeated_rows = int(duplicate_mask.sum())
            extra_rows = int(
                group.duplicated(
                    duplicate_columns,
                    keep="first"
                ).sum()
            )

            issue(
                contract,
                f"возможные дубли: {repeated_rows} строк "
                f"в группах повторов, из них {extra_rows} повторных"
            )

    registry = pd.read_csv(
        REGISTRY_FILE,
        dtype={"regnum": "string"},
        encoding="utf-8-sig"
    )

    if registry["regnum"].duplicated().any():
        raise ValueError("В реестре повторяются regnum")

    for contract in contracts - set(registry["regnum"]):
        issue(contract, "нет в реестре")

    for contract in contracts - set(positions["regnum"]):
        issue(contract, "нет распознанных позиций")

    totals = registry.set_index("regnum")["amount_rur"]

    for contract, group in positions.groupby("regnum"):
        if (
            contract in totals.index
            and group["line_amount_used"].notna().all()
        ):
            total = group["line_amount_used"].sum()
            expected = float(totals.loc[contract])

            if abs(total - expected) > 1:
                issue(
                    contract,
                    f"сумма позиций {total:.2f}, "
                    f"сумма контракта {expected:.2f}"
                )

    result = (
        pd.DataFrame({"regnum": sorted(contracts)})
        .merge(
            registry,
            on="regnum",
            how="left",
            validate="one_to_one"
        )
        .merge(
            positions,
            on="regnum",
            how="left",
            validate="one_to_many",
            suffixes=("_registry", "")
        )
    )

    result["target_in_scope"] = ""
    result["positions_complete"] = ""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("issue contracts:")

    if not issues:
        print("нет")

    for contract, messages in sorted(issues.items()):
        print(f"{contract}: {'; '.join(sorted(messages))}")

    print(f"\nКонтрактов: {len(contracts)}")
    print(f"Позиций: {len(positions)}")
    print(f"Сохранено: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
