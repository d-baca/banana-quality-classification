import pandas as pd


def load_csv_data(file_path: str = "data/banana_quality.csv") -> pd.DataFrame:
    """
    Wczytuje CSV z danymi numerycznymi o bananach, ustawia poprawne nazwy kolumn, zwraca DataFrame.
    W przypadku błędu zwraca pusty DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        data["Quality"] = data["Quality"].astype("category")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    df = load_csv_data()
    if not df.empty:
        print(df.shape)
    else:
        print("No data to display.")
