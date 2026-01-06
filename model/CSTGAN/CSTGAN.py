from typing import Union, Optional
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from model.CSTGAN.Discriminator import SpatialDiscriminator, TemporalDiscriminator, gradient_penalty
from model.CSTGAN.Generator import Generator
from model.BaseNNImputer import BaseNNImputer
from processUtil import denormalize_torch, logger, setup_logger, cal_mae, cal_rmse, cal_bias, cal_mre, cal_r

class CSTGAN(BaseNNImputer):
    """The PyTorch implementation of the USGAN model."""
    
    def __init__(
        self,
        seq_len: int,
        img_size: int,
        rnn_hidden_size: int,
        lambda_gp: float = 10,
        lambda_adv: float = 1e-3,
        epochs: int = 100,
        patience: Optional[int] = None,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        trained_model_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "better",
    ):
        super().__init__(
            epochs, patience, 0, device, 
            saving_path, model_saving_strategy
        )
        self._setup_device(device)
        self.model = Generator(
            seq_len=seq_len, img_size=img_size, 
            rnn_hidden_size=rnn_hidden_size, device=self.device
            ).to(self.device)
        self.Ds  = SpatialDiscriminator().to(self.device)
        self.Dt = TemporalDiscriminator(seq_len).to(self.device)
        self.opt_G = optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.opt_Ds = optim.Adam(self.Ds.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.opt_Dt = optim.Adam(self.Dt.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.lambda_gp  = lambda_gp
        self.lambda_adv =lambda_adv
        self.H = img_size
        self.W = img_size
        
        if trained_model_path is not None:
            self.load(trained_model_path)     
        self._send_model_to_given_device()
        self._print_model_size()

    def _assemble_input_for_training(self, data: list) -> dict:
        """Assemble input for training."""
        # Fetch data
        (
            indices, X, X_mask, missing_mask, deltas,
            back_X, back_X_mask, back_missing_mask, back_deltas,
        ) = self._send_data_to_given_device(data)
        
        # Assemble input data
        inputs = {
            "indices": indices,
            "forward": {
                "X": X,
                "X_mask": X_mask,
                "missing_mask": missing_mask,
                "deltas": deltas,
            },
            "backward": {
                "X": back_X,
                "X_mask": back_X_mask,
                "missing_mask": back_missing_mask,
                "deltas": back_deltas,
            },
        }
        
        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        """Assemble input for testing."""
        return self._assemble_input_for_training(data)

    def _train_model(
        self,
        dataset,
    ) -> None:
        training_loader, val_loader = dataset.get_train_val_dataloader()
        self.best_loss = float("inf")
        self.best_model_dict = None
        val_true = dataset.X_val_raw
        if val_true is not None:
            val_true = torch.from_numpy(val_true).to(self.device)
            mean_val = torch.tensor(dataset.mean_val).to(self.device)
            std_val = torch.tensor(dataset.std_val).to(self.device)
        
        try:
            training_step = 0
            train_loss_G_collector = []
            train_adv_loss_collector = []
            train_loss_D_collector = []
            train_reconstruction_MAE_collector = []
            
            for epoch in range(self.epochs):
                train_bar = tqdm(training_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
                
                for idx, data in enumerate(train_bar):
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)

                    # -------------------------
                    # 1. Train Discriminators
                    # -------------------------
                    for _ in range(1):
                    ## generate fake images by Generator
                        with torch.no_grad():
                            fake = self.model(inputs)["imputed_data"]

                        ## real images and mask
                        real = inputs["forward"]["X"]
                        mask = inputs["forward"]["missing_mask"]

                        B, T, HW = real.shape
                        loss_Ds, gp = 0, 0
                        for t in range(T):
                            real_t = real[:, t, :].view(B, 1, self.H, self.W)
                            fake_t = fake[:, t, :].view(B, 1, self.H, self.W)
                            mask_t = mask[:, t, :].view(B, 1, self.H, self.W)

                            loss_Ds += self.Ds(fake_t, mask_t).mean() \
                                    - self.Ds(real_t, mask_t).mean()
                            gp += gradient_penalty(self.Ds, real_t, fake_t, mask_t, self.device)
                        loss_Ds = loss_Ds / T + self.lambda_gp * gp / T

                        loss_Dt = self.Dt(fake.view(B, T, -1),
                            mask.view(B, T, -1)).mean() - self.Dt(
                            real.view(B, T, -1),
                            mask.view(B, T, -1)).mean()
                        self.opt_Ds.zero_grad()
                        self.opt_Dt.zero_grad()
                        (loss_Ds + loss_Dt).backward()
                        self.opt_Ds.step()
                        self.opt_Dt.step()
                    # -------------------------
                    # 2. Train Generator
                    # -------------------------
                    ret = self.model(inputs)
                    fake = ret["imputed_data"]
                    mask = inputs["forward"]["missing_mask"]

                    adv_loss = 0
                    for t in range(T):
                        adv_loss += -self.Ds(
                            fake[:, t, :].view(B, 1, self.H, self.W),
                            mask[:, t, :].view(B, 1, self.H, self.W)
                        ).mean()

                    adv_loss += -self.Dt(
                        fake.view(B, T, -1),
                        mask.view(B, T, -1)
                    ).mean()
                    adv_loss = adv_loss / (T + 1)

                    total_loss = ret["loss"] + self.lambda_adv * adv_loss

                    self.opt_G.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.opt_G.step()

                    train_loss_G_collector.append(total_loss.item())
                    train_adv_loss_collector.append(adv_loss.item())
                    train_loss_D_collector.append(loss_Ds.item() + loss_Dt.item())
                    train_reconstruction_MAE_collector.append(
                        ret.get("reconstruction_MAE", torch.tensor(0.0)).sum().item()
                    )
                    
                    # if epoch % 5 == 0:
                    #     print(
                    #         f"[{epoch}] "
                    #         f"G_loss={total_loss.item():.4f} "
                    #         f"Adv={adv_loss.item():.4f} "
                    #         f"D_s={loss_Ds.item():.4f} "
                    #         f"D_t={loss_Dt.item():.4f}"
                    #     )
                
                train_G_loss = np.mean(train_loss_G_collector)
                train_D_loss = np.mean( train_loss_D_collector)
                train_adv_loss = np.mean( train_adv_loss_collector)
                
                train_reconstruction_MAE = np.mean(train_reconstruction_MAE_collector)
                
                if val_loader is not None:
                    self.model.eval()
                    epoch_val_loss_G_collector = []
                    imputation_collector, mask_collector = [], []
                    
                    with torch.no_grad():
                        for data in val_loader:
                            inputs = self._assemble_input_for_validating(data)
                            ret = self.model(inputs)
                            imputed_data = ret["imputed_data"]       

                            imputation_collector.append(imputed_data)
                            mask_collector.append(inputs['forward']['missing_mask'])
                            epoch_val_loss_G_collector.append(ret["loss"].sum().item())
                    
                    mean_val_G_loss = np.mean(epoch_val_loss_G_collector)
                    imputation_collector = torch.cat(imputation_collector)
                    mask_collector = torch.cat(mask_collector)
                    
                    mean_val_bias, mean_val_mae, mean_val_rmse, mean_val_r, mean_val_mre = torch.nan, torch.nan, torch.nan, torch.nan, torch.nan
                    if val_true is not None:
                        mask_collector = mask_collector > 0
                        inv_mask_collector = torch.logical_not(mask_collector)
                        imputation_collector = denormalize_torch(imputation_collector, mean_val, std_val)
                        mean_val_bias = cal_bias(imputation_collector, val_true, inv_mask_collector)
                        mean_val_mae = cal_mae(imputation_collector, val_true, inv_mask_collector)
                        mean_val_rmse = cal_rmse(imputation_collector, val_true, inv_mask_collector)
                        mean_val_r = cal_r(imputation_collector, val_true, inv_mask_collector)
                        mean_val_mre = cal_mre(imputation_collector, val_true, inv_mask_collector)
                    
                    # Save validating loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "generation_loss": mean_val_G_loss,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_loss_dict)
                    
                    logger.info(
                        f"Epoch {epoch} - "
                        f"train G loss: {train_G_loss:.4f}, "
                        f"train Adv loss: {train_adv_loss:.4f}, "
                        f"train D loss: {train_D_loss:.4f}, "
                        f"train reconstruction MAE: {train_reconstruction_MAE:.4f}, "
                        f"validate loss: {mean_val_G_loss:.4f}, "
                        f"validate bias: {mean_val_bias:.4f}mm, "
                        f"validate R: {mean_val_r:.4f}, "
                        f"validate rmse: {mean_val_rmse:.4f}mm"
                    )
                    
                    mean_loss = mean_val_G_loss
                else:
                    logger.info(
                        f"Epoch {epoch} - "
                        f"training G loss: {train_G_loss:.4f}, "
                        f"training D loss: {train_D_loss:.4f}, "
                        f"training reconstruction MAE: {train_reconstruction_MAE:.4f}"
                    )
                    mean_val_mae = 0
                
                if np.isnan(mean_loss):
                    logger.warning(
                        f"Attention: got NaN loss in Epoch {epoch}. This may lead to unexpected errors."
                    )
                
                if mean_val_mae < self.best_loss:
                    self.best_loss = mean_val_mae
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                    
                    # Save the model if necessary
                    self._auto_save_model_if_necessary(
                        training_finished=False,
                        saving_name=f"{self.__class__.__name__}_epoch{epoch}",
                    )
                else:
                    self.patience -= 1
                
                if self.patience == 0:
                    logger.info(
                        "Exceeded the training patience. Terminating the training procedure..."
                    )
                    break
                    
        except Exception as e:
            logger.error(f"Exception: {e}")
            
            if self.best_model_dict is None:
                raise RuntimeError(
                    "Training got interrupted. Model was not trained. "
                    "Please investigate the error printed above."
                )
            else:
                logger.warning(
                    "Training got interrupted. Please investigate the error printed above.\n"
                    "Model got trained and will load the best checkpoint so far for testing.\n"
                    "If you don't want it, please try fit() again."
                )
        
        if torch.isnan(self.best_loss):
            raise ValueError("Something is wrong. best_loss is NaN after training.")
        
        logger.info("Finished training.")

    def train(self, dataset) -> None:

        self._train_model(dataset)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # Set the model as eval status to freeze it.
        
        # Step 3: Save the model if necessary
        self._auto_save_model_if_necessary(training_finished=True)
    
    def predict(self, dataset) -> np.ndarray:
        test_loader = dataset.get_test_dataloader()
        test_true = dataset.X_test_raw
        if test_true is not None:
            test_true = torch.from_numpy(test_true).to(self.device)
            mean_test = torch.tensor(dataset.mean_test).to(self.device)
            std_test = torch.tensor(dataset.std_test).to(self.device)
        
        mask_collector, imputation_collector = [], []
        self.model.eval()
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Predicting"):
                inputs = self._assemble_input_for_testing(data)
                ret = self.model(inputs)
                imputed_data = ret["imputed_data"]       

                imputation_collector.append(imputed_data)
                mask_collector.append(inputs['forward']['missing_mask'])
        
        imputation_collector = torch.cat(imputation_collector)
        mask_collector = torch.cat(mask_collector)
        
        test_bias, test_mae, test_rmse, test_r, test_mre = torch.nan, torch.nan, torch.nan, torch.nan, torch.nan
        if test_true is not None:
            mask_collector = mask_collector > 0
            inv_mask_collector = torch.logical_not(mask_collector)
            imputation_collector = denormalize_torch(imputation_collector, mean_test, std_test)
            test_bias = cal_bias(imputation_collector, test_true, inv_mask_collector)
            test_mae = cal_mae(imputation_collector, test_true, inv_mask_collector)
            test_rmse = cal_rmse(imputation_collector, test_true, inv_mask_collector)
            test_r = cal_r(imputation_collector, test_true, inv_mask_collector)
            test_mre = cal_mre(imputation_collector, test_true, inv_mask_collector)
        imputation_collector = imputation_collector.cpu().numpy()
        return imputation_collector, test_r.cpu().numpy(), test_bias.cpu().numpy(), test_mae.cpu().numpy(), test_rmse.cpu().numpy(), test_mre.cpu().numpy()
    