import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess_stroke(file_path):
    print("=" * 60)
    print("MEMUAT DATASET")
    print("=" * 60)

    try:
        df = pd.read_csv(file_path)

        print("Dataset berhasil dimuat")
        print(f"Shape Dataset : {df.shape}")

    except FileNotFoundError:
        print(f"File {file_path} tidak ditemukan")
        return None

    except Exception as e:
        print(f"Terjadi error : {e}")
        return None

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\nInfo Dataset")
    print(df.info())

    print("\n5 Baris Pertama")
    print(df.head())

    print("\nStatistik Deskriptif")
    print(df.describe())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nJumlah Duplikat")
    print(df.duplicated().sum())

    print("\nDistribusi Target")
    print(df['stroke'].value_counts())

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Missing Value BMI
    if df['bmi'].isnull().sum() > 0:

        median_bmi = df['bmi'].median()

        df['bmi'].fillna(
            median_bmi,
            inplace=True
        )

        print(
            f"Missing value bmi diisi median ({median_bmi:.2f})"
        )

    # Hapus Duplikat
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print(f"Duplikat dihapus : {before - after}")

    # Encoding
    categorical_cols = [
        'gender',
        'ever_married',
        'work_type',
        'Residence_type',
        'smoking_status'
    ]

    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])
        print(f"Encoding : {col}")

    # Outlier Handling
    print("\nMenangani Outlier Dengan IQR")
    numerical_cols = ['age','avg_glucose_level','bmi']

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - (1.5 * IQR)
        upper = Q3 + (1.5 * IQR)

        before_outlier = (
            ((df[col] < lower) |
             (df[col] > upper))
        ).sum()

        df[col] = np.where(
            df[col] < lower,
            lower,
            np.where(
                df[col] > upper,
                upper,
                df[col]
            )
        )

        after_outlier = (((df[col] < lower)|(df[col] > upper))).sum()

        print(f"{col} : sebelum={before_outlier}, sesudah={after_outlier}")

    # Scaling
    print("\nStandard Scaling")

    feature_cols = df.drop('stroke',axis=1).columns
    scaler = StandardScaler()

    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("Scaling selesai")

    print("\n" + "=" * 60)
    print("MENYIMPAN HASIL")
    print("=" * 60)

    output_file = "stroke_preprocessing.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print(f"Dataset berhasil disimpan ke {output_file}")
    print(f"Shape akhir : {df.shape}")

    print("=" * 60)
    print("DATA SIAP DILATIH")
    print("=" * 60)

    return df


if __name__ == "__main__":
    processed_df = preprocess_stroke("../healthcare-dataset-stroke-data.csv")

    if processed_df is not None:
        print("\nPreview Data")
        print(processed_df.head())