"""
automate_Louis.py
Otomatisasi preprocessing dataset cancer patient.
Konversi dari Eksperimen_Louis.ipynb (Section 5).
"""

import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

KOLOM_IDENTIFIER = ["index", "Patient Id"]
URUTAN_LEVEL = [["Low", "Medium", "High"]]


def preprocess_data(input_path, output_path=None):
    # Baca dataset
    df = pd.read_csv(input_path)

    # 1. Buang kolom identifier (bukan fitur prediktif)
    df = df.drop(columns=KOLOM_IDENTIFIER)

    # 2. Encode target: Low=0, Medium=1, High=2
    encoder = OrdinalEncoder(categories=URUTAN_LEVEL)
    df["Level"] = encoder.fit_transform(df[["Level"]]).astype(int)

    # 3. Simpan hasil bila diminta
    if output_path:
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(BASE_DIR, "..", "Cancer_Raw.csv")
    output_path = os.path.join(BASE_DIR, "cancer_preprocessing.csv")

    df_clean = preprocess_data(input_path, output_path)
    print("Preprocessing selesai. Shape:", df_clean.shape)
    print("Distribusi Level:", df_clean["Level"].value_counts().to_dict())