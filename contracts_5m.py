import os
import pandas as pd


CONTRACT_ID_COL = 'regnum'
STRATUM_COL = 'sampling_stratum'
GT_5M_LABEL = 'gt_5m'


def export_gt_5m_contract_numbers(
    input_file='output/relevant_contracts_all.csv',
    output_file='output/selected_contract_numbers_gt_5m.csv'
):
    df = pd.read_csv(
        input_file,
        dtype={CONTRACT_ID_COL: 'string'}
    )

    gt_5m_contracts = df.loc[
        df[STRATUM_COL].eq(GT_5M_LABEL),
        [CONTRACT_ID_COL]
    ].copy()

    gt_5m_contracts = (
        gt_5m_contracts
        .dropna(subset=[CONTRACT_ID_COL])
        .drop_duplicates()
        .rename(columns={CONTRACT_ID_COL: 'contract_number'})
        .sort_values('contract_number')
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    gt_5m_contracts.to_csv(
        output_file,
        index=False,
        encoding='utf-8-sig'
    )

    print(
        'Сохранён список номеров контрактов дороже 5 млн:',
        len(gt_5m_contracts)
    )


if __name__ == '__main__':
    export_gt_5m_contract_numbers()
