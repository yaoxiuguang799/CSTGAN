import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union, Optional, Iterable, Dict
import torch

# Import modules that should be available in the package structure
# Note: These imports may need to be adjusted based on your actual project structure

from processUtil import logger, setup_logger


class BaseNNImputer(ABC):
    """
    The abstract base class for all neural-network imputation models.
    
    This class combines the functionality of BaseModel, BaseNNModel, and the original BaseNNImputer.
    
    Parameters
    ----------
    batch_size : int
        Size of the batch input into the model for one step.
    epochs : int
        Training epochs, i.e. the maximum rounds of the model to be trained with.
    patience : int, optional
        Number of epochs the training procedure will keep if loss doesn't decrease.
        Once exceeding the number, the training will stop.
        Must be smaller than or equal to the value of `epochs`.
    num_workers : int, default = 0
        The number of subprocesses to use for data loading.
        0 means data loading will be in the main process, i.e. there won't be subprocesses.
    device : Union[str, torch.device, list], optional
        The device for the model to run on. It can be a string, a :class:`torch.device` object,
        or a list of them. If not given, will try to use CUDA devices first
        (will use the default CUDA device if there are multiple), then CPUs.
    saving_path : str, optional
        The path for automatically saving model checkpoints and tensorboard files.
        Will not save if not given.
    model_saving_strategy : str, optional, default = "best"
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever
        the model performs better than in previous epochs.
    
    Attributes
    ----------
    model : object
        The underlying model or algorithm to finish the task.
    best_model_dict : dict
        A dictionary contains the trained model that achieves the best performance.
    best_loss : float
        The criteria to judge whether the model's performance is the best so far.
    summary_writer : None or torch.utils.tensorboard.SummaryWriter
        The event writer to save training logs.
    """
    
    def __init__(
        self,
        epochs: int,
        patience: Optional[int] = None,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
    ):
        # Validate model saving strategy
        saving_strategies = [None, "best", "better"]
        assert model_saving_strategy in saving_strategies, (
            f"model_saving_strategy must be one of {saving_strategies}, "
            f"but got {model_saving_strategy}."
        )
        
        # Validate patience parameter
        if patience is None:
            patience = -1  # Early stopping on patience won't work if it is set as < 0
        else:
            assert patience <= epochs, (
                f"patience must be smaller than epochs which is {epochs}, "
                f"but got patience={patience}"
            )
        
        # Set up training hyper-parameters
        self.epochs = epochs
        self.patience = patience
        self.original_patience = patience
        self.num_workers = num_workers
        self.model_saving_strategy = model_saving_strategy
        
        # Initialize model and optimizer
        self.model = None
        self.optimizer = None
        self.best_model_dict = None
        self.best_loss = float("inf")
        
        # Set up device and paths
        self.device = None
        self.saving_path = None
        self.summary_writer = None
        
        self._setup_device(device)
        self._setup_path(saving_path)
    
    def _setup_device(self, device: Union[None, str, torch.device, list]) -> None:
        """Set up the device for model training and inference."""
        if device is None:
            # Use the first cuda device if available, otherwise use cpu
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            logger.info(f"No given device, using default device: {self.device}")
        else:
            if isinstance(device, str):
                self.device = torch.device(device.lower())
            elif isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, list):
                if len(device) == 0:
                    raise ValueError(
                        "The list of devices should have at least 1 device, but got 0."
                    )
                elif len(device) == 1:
                    self._setup_device(device[0])
                    return
                
                # Parallel training on multiple CUDA devices
                device_list = []
                for idx, d in enumerate(device):
                    if isinstance(d, str):
                        d = d.lower()
                        assert "cuda" in d, (
                            "The feature of training on multiple devices currently "
                            "only supports CUDA devices."
                        )
                        device_list.append(torch.device(d))
                    elif isinstance(d, torch.device):
                        assert "cuda" in d.type, (
                            "The feature of training on multiple devices currently "
                            "only supports CUDA devices."
                        )
                        device_list.append(d)
                    else:
                        raise TypeError(
                            f"Devices in the list should be str or torch.device, "
                            f"but the device with index {idx} is {type(d)}."
                        )
                
                if len(device_list) > 1:
                    self.device = device_list
                else:
                    self.device = device_list[0]
            else:
                raise TypeError(
                    f"device should be str/torch.device/a list containing str or torch.device, "
                    f"but got {type(device)}"
                )
        
        # Check CUDA availability if using CUDA
        if (isinstance(self.device, list) and "cuda" in self.device[0].type) or \
           (isinstance(self.device, torch.device) and "cuda" in self.device.type):
            assert torch.cuda.is_available() and torch.cuda.device_count() > 0, (
                "You are trying to use CUDA for model training, "
                "but CUDA is not available in your environment."
            )
    
    def _setup_path(self, saving_path: Optional[str]) -> None:
        """Set up the path for saving models and logs."""
        if isinstance(saving_path, str):
            # Get the current time to append to saving_path
            time_now = datetime.now().strftime("%Y%m%d_T%H%M%S")
            # The actual saving_path for saving both the best model and the tensorboard file
            self.saving_path = saving_path
            if not os.path.exists(self.saving_path):
                os.makedirs(self.saving_path)
            
            # Initialize logger
            time_now_log = datetime.now().strftime("%Y-%m-%d_T%H-%M-%S")
            log_name = f"{self.__class__.__name__}_{time_now_log}.log"
            log_saving_path = os.path.join(self.saving_path, log_name)
            setup_logger(log_saving_path, log_name, 'w')
            
            logger.info(f"Model files will be saved to {self.saving_path}")
            logger.info(f"Tensorboard file will be saved to {log_saving_path}")
        else:
            logger.warning(
                "saving_path not given. Model files and tensorboard file will not be saved."
            )
    
    def _send_model_to_given_device(self) -> None:
        """Send the model to the specified device."""
        if isinstance(self.device, list):
            # Parallel training on multiple devices
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device)
            self.model = self.model.cuda()
            logger.info(
                f"Model has been allocated to the given multiple devices: {self.device}"
            )
        else:
            self.model = self.model.to(self.device)
    
    def _send_data_to_given_device(self, data: Iterable) -> Iterable:
        """Send data to the specified device."""
        if isinstance(self.device, torch.device):
            # Single device
            data = map(lambda x: x.to(self.device), data)
        else:
            # Parallel training on multiple devices
            data = map(lambda x: x.cuda(), data)
        return data
    
    def _save_log_into_tb_file(
        self, 
        step: int, 
        stage: str, 
        loss_dict: Dict[str, float]
    ) -> None:
        """Save training logs into the tensorboard file."""
        if self.summary_writer is not None:
            for item_name, loss in loss_dict.items():
                if any(keyword in item_name for keyword in ["loss", "MAE", "MSE", "RMSE"]):
                    self.summary_writer.add_scalar(f"{stage}/{item_name}", loss, step)
    
    def _auto_save_model_if_necessary(
        self, 
        training_finished: bool = True, 
        saving_name: Optional[str] = None
    ) -> None:
        """Automatically save the current model into a file if in need."""
        if self.saving_path is not None and self.model_saving_strategy is not None:
            name = self.__class__.__name__ if saving_name is None else saving_name
            saving_path = os.path.join(self.saving_path, name)
            
            if not training_finished and self.model_saving_strategy == "better":
                self.save(saving_path)
            elif training_finished and self.model_saving_strategy == "best":
                self.save(saving_path)
    
    def save(self, saving_path: str, overwrite: bool = False) -> None:
        """Save the model with current parameters to a disk file."""
        # Split the saving dir and file name from the given path
        saving_dir, file_name = os.path.split(saving_path)
        
        if not file_name.endswith(".pt"):
            file_name += ".pt"
        
        # Rejoin the path for saving the model
        saving_path = os.path.join(saving_dir, file_name)
        
        if os.path.exists(saving_path):
            if overwrite:
                logger.warning(
                    f"File {saving_path} exists. Argument overwrite is True. Overwriting now..."
                )
            else:
                logger.error(f"File {saving_path} exists. Saving operation aborted.")
                return
        
        try:
            os.makedirs(saving_dir, exist_ok=True)
            torch.save(self.model.state_dict(), saving_path)
            logger.info(f"Saved the model to {saving_path}.")
        except Exception as e:
            raise RuntimeError(
                f'Failed to save the model to "{saving_path}" because of the below error! \n{e}'
            )
    
    def load(self, path: str) -> None:
        assert os.path.exists(path), f"Model file {path} does not exist."

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        try:
            state_dict = torch.load(path, map_location=device)
            self.model.to(device)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()

            logger.info(
                f"Model loaded successfully from {path} on device {device}."
            )
        except Exception as e:
            raise RuntimeError(
                f'Failed to load the model from "{path}".\n{e}'
            )
    
    def save_model(self, saving_path: str, overwrite: bool = False) -> None:
        """Deprecated: Use save() instead."""
        logger.warning(
            "DeprecationWarning: The method save_model is deprecated. Please use save() instead."
        )
        self.save(saving_path, overwrite)
    
    def load_model(self, path: str) -> None:
        """Deprecated: Use load() instead."""
        logger.warning(
            "DeprecationWarning: The method load_model is deprecated. Please use load() instead."
        )
        self.load(path)
    
    def _print_model_size(self) -> None:
        """Print the number of trainable parameters in the initialized NN model."""
        if self.model is None:
            logger.warning("Model has not been initialized yet.")
            return
        logger.info("Model parameter details:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(
                    f"{name:<50} | shape={tuple(param.shape)} | params={param.numel():,}"
                )
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Model initialized successfully with the number of trainable parameters: {num_params:,}"
        )

