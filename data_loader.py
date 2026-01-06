import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from processUtil import parse_delta


# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------
def read_h5(h5_file):
    with h5py.File(h5_file, "r") as hf:
        X_raw = hf["X_raw"][:]
        X = hf["X_norm"][:]
        mask = hf["mask"][:]
        mean = hf.attrs["mean"]
        std = hf.attrs["std"]
    return X_raw, X, mask, mean, std


def fill_with_last_observation(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values along axis=1."""
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return np.nan_to_num(out)  # fill remaining NaN as 0


class BaseDataset(Dataset):
    """
    BaseDataset now serves as the training dataset.
    No artificial missing is applied.
    """

    def __init__(self, X: np.ndarray, mask: np.ndarray):
        self.X = X
        self.mask = mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = np.copy(self.X[idx])
        missing_mask = np.copy(self.mask[idx])
        X = np.nan_to_num(X)
        X_mask = np.copy(X)
        X_mask[missing_mask==0] = 0.0

        forward = {
            "X": X,
            "X_mask": X_mask,
            "missing_mask": missing_mask,
            "deltas": parse_delta(missing_mask),
        }

        backward_mask = np.flip(missing_mask, axis=0).copy()
        backward = {
            "X": np.flip(X, axis=0).copy(),
            "X_mask": np.flip(X_mask, axis=0).copy(),
            "missing_mask": backward_mask,
            "deltas": parse_delta(backward_mask),
        }

        return (
            torch.tensor(idx, dtype=torch.long),
            torch.from_numpy(forward["X"]).float(),
            torch.from_numpy(forward["X_mask"]).float(),
            torch.from_numpy(forward["missing_mask"]).float(),
            torch.from_numpy(forward["deltas"]).float(),
            torch.from_numpy(backward["X"]).float(),
            torch.from_numpy(backward["X_mask"]).float(),
            torch.from_numpy(backward["missing_mask"]).float(),
            torch.from_numpy(backward["deltas"]).float(),
        )
    
class CSTGANDataLoader:
    def __init__(
        self,
        train_dataset_path: str,
        val_dataset_path: str = None,
        test_dataset_path: str = None,
        batch_size=32,
        num_workers=0,
        val_ratio=0.1,
        seed=42,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed

        # -------- load train file --------
        if train_dataset_path is not None and test_dataset_path is None:
            self.X_train_raw, self.X_train, self.mask_train, mean, std = read_h5(train_dataset_path)
            self.X_train_raw = None
            self.X_train = self._merge_last_two_dims(self.X_train)
            self.mask_train = self._merge_last_two_dims(self.mask_train)
        else:
            self.X_train_raw, self.X_train, self.mask_train, mean, std = None, None, None, 0, 0
        
        # -------- load train file --------
        if val_dataset_path is not None and test_dataset_path is None:
            self.X_val_raw, self.X_val, self.mask_val, self.mean_val, self.std_val = read_h5(val_dataset_path)
            self.X_val_raw = self._merge_last_two_dims(self.X_val_raw)
            self.X_val = self._merge_last_two_dims(self.X_val)
            self.mask_val = self._merge_last_two_dims(self.mask_val)
        else:
            self.X_val_raw, self.X_val, self.mask_val, self.mean_val, self.std_val = None, None, None, 0, 0
        
        # -------- load test file --------
        if  test_dataset_path is not None:
            self.X_test_raw, self.X_test, self.mask_test, self.mean_test, self.std_test = read_h5(test_dataset_path)
            self.X_test_raw = self._merge_last_two_dims(self.X_test_raw)
            self.X_test = self._merge_last_two_dims(self.X_test)
            self.mask_test = self._merge_last_two_dims(self.mask_test)
    # ---------------------------------
    @staticmethod
    def _merge_last_two_dims(arr):
        """Merge the last two dimensions into one."""
        return arr.reshape(*arr.shape[:-2], -1)
    
    def _build_loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def get_train_dataloader(self):
        return self._build_loader(
            BaseDataset(self.X_train,self.mask_train),
            shuffle=True,
        )

    def get_val_dataloader(self):
        return self._build_loader(
            BaseDataset(self.X_val,self.mask_val),
            shuffle=False,
        )

    def get_test_dataloader(self):
        return self._build_loader(
            BaseDataset(self.X_test,self.mask_test),
            shuffle=False,
        )

    def get_train_val_dataloader(self):
        return self.get_train_dataloader(), self.get_val_dataloader()

