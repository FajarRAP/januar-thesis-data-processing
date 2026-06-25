import pandas as pd

class KFoldCrossValidation:
    def __init__(self, df: pd.DataFrame, K: int):
        self.df = df
        self.K = K
        self.splits = []
        
        n_samples = df.shape[0]
        base_size = n_samples // K  # Ukuran dasar tiap fold (contoh: 161)
        remainder = n_samples % K   # Sisa data yang mau didistribusikan (contoh: 1)
        
        current_index = 0
        for i in range(K):
            # Jika masih ada 'remainder', tambahkan 1 data ekstra ke fold ini
            fold_size = base_size + (1 if i < remainder else 0)
            
            # Potong dataframe sesuai fold_size
            self.splits.append(df.iloc[current_index : current_index + fold_size])
            current_index += fold_size

    def get_train_test_split(self, fold_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        test_data = self.splits[fold_index]
        train_data = pd.concat([self.splits[i] for i in range(self.K) if i != fold_index], ignore_index=True)
        
        return train_data, test_data