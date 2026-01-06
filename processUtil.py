import numpy as np
import torch
from tsdb.utils.logging import Logger
from typing import Union, Optional
import logging

# initialize a logger for PyPOTS logging
logger_creator = Logger(name="PyPOTS running log")
logger = logger_creator.logger



def setup_logger(log_file_path, log_name, mode="a"):
    """set up log file
    mode : 'a'/'w' mean append/overwrite,
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False  # prevent the child logger from propagating log to the root logger (twice), not necessary
    return logger


def normalize_with_nan(x, mean, std):
    mask = (~np.isnan(x)).astype(np.float32)
    x_filled = np.nan_to_num(x, nan=mean)
    x_norm = (x_filled - mean) / (std + 1e-8)
    x_norm[mask == 0] = 0.0
    return x_norm

def denormalize_torch(x_norm, mean, std, mask=None, restore_nan=True):
    """
    Denormalize tensor with optional NaN restoration.

    Parameters
    ----------
    x_norm : torch.Tensor
        Normalized tensor
    mean : float or torch.Tensor
        Mean used in normalization (broadcastable)
    std : float or torch.Tensor
        Std used in normalization (broadcastable)
    mask : torch.Tensor or None
        1 for valid, 0 for missing (same shape as x_norm)
    restore_nan : bool
        Whether to restore NaN at missing locations

    Returns
    -------
    x : torch.Tensor
        Denormalized tensor
    """
    # Ensure tensor type/device consistency
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, device=x_norm.device, dtype=x_norm.dtype)
    if not torch.is_tensor(std):
        std = torch.tensor(std, device=x_norm.device, dtype=x_norm.dtype)

    x = x_norm * std + mean

    if mask is not None:
        if restore_nan:
            x = x.masked_fill(mask == 0, float('nan'))
        else:
            # Keep mean value at missing locations
            x = x * mask + mean * (1.0 - mask)

    return x

def cal_bias(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate Mean Absolute Error (MAE), ignoring NaN values."""

    assert isinstance(predictions, type(targets)), (
        f"types of inputs and target must match, but got "
        f"type(inputs)={type(predictions)}, type(target)={type(targets)}"
    )

    lib = np if isinstance(predictions, np.ndarray) else torch

    # convert list -> array/tensor
    if isinstance(predictions, list):
        predictions = lib.asarray(predictions) if lib is np else torch.tensor(predictions)
        targets = lib.asarray(targets) if lib is np else torch.tensor(targets)
        if masks is not None:
            masks = lib.asarray(masks) if lib is np else torch.tensor(masks)

    # build valid mask (exclude NaN)
    if lib is np:
        valid_mask = ~np.isnan(predictions) & ~np.isnan(targets)
    else:
        valid_mask = ~torch.isnan(predictions) & ~torch.isnan(targets)

    # combine with user-provided masks
    if masks is not None:
        valid_mask = valid_mask & (masks == 1)

    if lib.sum(valid_mask) == 0:
        return 0.0
    diff = predictions[valid_mask] - targets[valid_mask]
    return lib.mean(diff)

def cal_mae(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate Mean Absolute Error (MAE), ignoring NaN values."""

    assert isinstance(predictions, type(targets)), (
        f"types of inputs and target must match, but got "
        f"type(inputs)={type(predictions)}, type(target)={type(targets)}"
    )

    lib = np if isinstance(predictions, np.ndarray) else torch

    # convert list -> array/tensor
    if isinstance(predictions, list):
        predictions = lib.asarray(predictions) if lib is np else torch.tensor(predictions)
        targets = lib.asarray(targets) if lib is np else torch.tensor(targets)
        if masks is not None:
            masks = lib.asarray(masks) if lib is np else torch.tensor(masks)

    # build valid mask (exclude NaN)
    if lib is np:
        valid_mask = ~np.isnan(predictions) & ~np.isnan(targets)
    else:
        valid_mask = ~torch.isnan(predictions) & ~torch.isnan(targets)

    # combine with user-provided masks
    if masks is not None:
        valid_mask = valid_mask & (masks == 1)

    if lib.sum(valid_mask) == 0:
        return 0.0
    diff = lib.abs(predictions[valid_mask] - targets[valid_mask])
    return lib.mean(diff)

def cal_mse(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate Mean Squared Error (MSE), ignoring NaN values."""

    assert isinstance(predictions, type(targets)), (
        f"types of inputs and target must match, but got "
        f"type(inputs)={type(predictions)}, type(target)={type(targets)}"
    )

    lib = np if isinstance(predictions, np.ndarray) else torch

    # list -> ndarray / tensor
    if isinstance(predictions, list):
        predictions = lib.asarray(predictions) if lib is np else torch.tensor(predictions)
        targets = lib.asarray(targets) if lib is np else torch.tensor(targets)
        if masks is not None:
            masks = lib.asarray(masks) if lib is np else torch.tensor(masks)

    # valid mask (exclude NaN)
    if lib is np:
        valid_mask = ~np.isnan(predictions) & ~np.isnan(targets)
    else:
        valid_mask = ~torch.isnan(predictions) & ~torch.isnan(targets)

    # combine with user mask
    if masks is not None:
        valid_mask = valid_mask & (masks == 1)

    if lib.sum(valid_mask) == 0:
        return 0.0

    diff = predictions[valid_mask] - targets[valid_mask]
    return lib.mean(diff ** 2)


def cal_rmse(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate Root Mean Squared Error (RMSE), ignoring NaN values."""

    # cal_mse already handles:
    # - numpy / torch / list
    # - NaN exclusion
    # - mask logic
    mse = cal_mse(predictions, targets, masks)

    lib = np if isinstance(mse, np.ndarray) or isinstance(mse, float) else torch
    return lib.sqrt(mse)

def cal_mre(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate Mean Absolute Error (MAE), ignoring NaN values."""

    assert isinstance(predictions, type(targets)), (
        f"types of inputs and target must match, but got "
        f"type(inputs)={type(predictions)}, type(target)={type(targets)}"
    )

    lib = np if isinstance(predictions, np.ndarray) else torch

    # convert list -> array/tensor
    if isinstance(predictions, list):
        predictions = lib.asarray(predictions) if lib is np else torch.tensor(predictions)
        targets = lib.asarray(targets) if lib is np else torch.tensor(targets)
        if masks is not None:
            masks = lib.asarray(masks) if lib is np else torch.tensor(masks)

    # build valid mask (exclude NaN)
    if lib is np:
        valid_mask = ~np.isnan(predictions) & ~np.isnan(targets)
    else:
        valid_mask = ~torch.isnan(predictions) & ~torch.isnan(targets)

    # combine with user-provided masks
    if masks is not None:
        valid_mask = valid_mask & (masks == 1)

    if lib.sum(valid_mask) == 0:
        return 0.0
    up = lib.mean(lib.abs(predictions[valid_mask] - targets[valid_mask]))
    down = lib.mean(lib.abs(targets[valid_mask]))
    return up/down
def cal_r(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate Pearson Correlation Coefficient (R), ignoring NaN values."""

    assert isinstance(predictions, type(targets)), (
        f"types of inputs and target must match, but got "
        f"type(inputs)={type(predictions)}, type(target)={type(targets)}"
    )

    lib = np if isinstance(predictions, np.ndarray) else torch

    # list -> array / tensor
    if isinstance(predictions, list):
        predictions = lib.asarray(predictions) if lib is np else torch.tensor(predictions)
        targets = lib.asarray(targets) if lib is np else torch.tensor(targets)
        if masks is not None:
            masks = lib.asarray(masks) if lib is np else torch.tensor(masks)

    # valid mask (exclude NaN)
    if lib is np:
        valid_mask = ~np.isnan(predictions) & ~np.isnan(targets)
    else:
        valid_mask = ~torch.isnan(predictions) & ~torch.isnan(targets)

    # combine with user mask
    if masks is not None:
        valid_mask = valid_mask & (masks == 1)

    if lib.sum(valid_mask) == 0:
        return 0.0

    x = predictions[valid_mask]
    y = targets[valid_mask]

    x_mean = lib.mean(x)
    y_mean = lib.mean(y)

    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = lib.sum(x_centered * y_centered)
    denominator = lib.sqrt(
        lib.sum(x_centered ** 2) * lib.sum(y_centered ** 2)
    ) + 1e-12

    return numerator / denominator


def _parse_delta_torch(missing_mask: torch.Tensor) -> torch.Tensor:
    """Generate the time-gap matrix (i.e. the delta metrix) from the missing mask.
    Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask : shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing data (0 means missing values, 1 means observed values).

    Returns
    -------
    delta :
        The delta matrix indicates the time gaps between observed values.
        With the same shape of missing_mask.


    """

    def cal_delta_for_single_sample(mask: torch.Tensor) -> torch.Tensor:
        """calculate single sample's delta. The sample's shape is [n_steps, n_features]."""
        d = []
        for step in range(n_steps):
            if step == 0:
                d.append(torch.zeros(1, n_features, device=device))
            else:
                d.append(
                    torch.ones(1, n_features, device=device) + (1 - mask[step]) * d[-1]
                )
        d = torch.concat(d, dim=0)
        return d

    device = missing_mask.device
    if len(missing_mask.shape) == 2:
        n_steps, n_features = missing_mask.shape
        delta = cal_delta_for_single_sample(missing_mask)
    else:
        n_samples, n_steps, n_features = missing_mask.shape
        delta_collector = []
        for m_mask in missing_mask:
            delta = cal_delta_for_single_sample(m_mask)
            delta_collector.append(delta.unsqueeze(0))
        delta = torch.concat(delta_collector, dim=0)

    return delta


def _parse_delta_numpy(missing_mask: np.ndarray) -> np.ndarray:
    """Generate the time-gap matrix (i.e. the delta metrix) from the missing mask.
    Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask : shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing data (0 means missing values, 1 means observed values).

    Returns
    -------
    delta :
        The delta matrix indicates the time gaps between observed values.
        With the same shape of missing_mask.

    """

    def cal_delta_for_single_sample(mask: np.ndarray) -> np.ndarray:
        """calculate single sample's delta. The sample's shape is [n_steps, n_features]."""
        d = []
        for step in range(seq_len):
            if step == 0:
                d.append(np.zeros(n_features))
            else:
                d.append(np.ones(n_features) + (1 - mask[step]) * d[-1])
        d = np.asarray(d)
        return d

    if len(missing_mask.shape) == 2:
        seq_len, n_features = missing_mask.shape
        delta = cal_delta_for_single_sample(missing_mask)
    else:
        n_samples, seq_len, n_features = missing_mask.shape
        delta_collector = []
        for m_mask in missing_mask:
            delta = cal_delta_for_single_sample(m_mask)
            delta_collector.append(delta)
        delta = np.asarray(delta_collector)
    return delta


def parse_delta(
    missing_mask: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Generate the time-gap matrix (i.e. the delta metrix) from the missing mask.
    Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask : shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing data (0 means missing values, 1 means observed values).

    Returns
    -------
    delta :
        The delta matrix indicates the time gaps between observed values.
        With the same shape of missing_mask.


    """
    if isinstance(missing_mask, np.ndarray):
        delta = _parse_delta_numpy(missing_mask)
    elif isinstance(missing_mask, torch.Tensor):
        delta = _parse_delta_torch(missing_mask)
    else:
        raise RuntimeError
    return delta