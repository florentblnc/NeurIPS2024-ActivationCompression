from pytorch_lightning.callbacks import Callback
import time

class LogActivationMemoryCallback(Callback):
    """
        Callback to log activation memory usage during training and validation.

        Args:
            log_activation_mem (bool): If True, logs activation memory for specific methods (SVD, HOSVD).
    """

    def __init__(self, log_activation_mem=False):
        self.log_activation_mem             = log_activation_mem    # if True: Log estimation of activation memory
        self.first_train_batch_logged       = False                 # a flag indicating that training of the 1st batch of the 1st epoch is finish
        self.training_begin                 = False                 # a flag indicating the beginning of training
        self.num_train_batches              = None                  # number of batch of data for training
        # self.num_val_batches                = None                  # number of batch of data for validating

    def on_train_epoch_start(self, trainer, model):
        """
        Called at the beginning of a training epoch.
        Attaches a list to store memory information to the model (for SVD and HOSVD) if logging is enabled and it is the first epoch.
        """
        if not self.training_begin:
            self.training_begin = True
            if self.log_activation_mem:
                model.attach_memory_info_list_HOSVD_SVD()
        

    def on_train_epoch_end(self, trainer, model):
        """
        Called at the end of a training epoch.
        Resets the attached memory list sizes for SVD or HOSVD methods after each epoch.
        """
        if self.log_activation_mem:
            if model.with_SVD: 
                model.get_activation_size_svd(self.num_train_batches) # Decomposition only occurs during training 
                model.reset_svd_size()
            elif model.with_HOSVD:
                model.get_activation_size_hosvd(self.num_train_batches) # Decomposition only occurs during training 
                model.reset_k_hosvd()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx):
        """
        Called at the end of a training batch.
        Logs activation memory for the first batch if applicable (for Vanilla Training and Gradient Filter)
        """
        self.num_train_batches = batch_idx + 1
        if self.log_activation_mem:
            if model.with_base or (hasattr(model, 'with_grad_filter') and model.with_grad_filter):
                if not self.first_train_batch_logged: # Log in the first epoch with the first train batch because the activation memory of these methods is stable.
                    model.get_activation_size(batch['image'])
                    self.first_train_batch_logged = True        

    # def on_validation_epoch_start(self, trainer, model):
    #     """
    #     Called at the beginning of a validation epoch (After training phase is ended)
    #     """

    # def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx):
    #     """
    #     Called at the end of a validation batch.
    #     """
    #     self.num_val_batches = batch_idx + 1

    # def on_train_end(self, trainer, model):
    #     """Finish training process"""