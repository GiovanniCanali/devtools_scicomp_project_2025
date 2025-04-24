import os


class EarlyStopping:
    """
    EarlyStopping class to stop training when validation loss does not improve.
    This class monitors the validation loss and stops training if it does not
    improve for a specified number of steps (patience).
    """

    def __init__(self, patience, path):
        """ """
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        if patience == 0:
            self.__call__ = lambda *args, **kwargs: None
        self.model_path = os.path.join(path, "best_model.pt")

    def __call__(self, val_loss):
        """
        Call method to check if the validation loss has improved.
        :param float val_loss: The current validation loss.
        :return: True if the validation loss has improved, False otherwise.
        :rtype: bool
        """
        if val_loss < self.best_loss:
            print(
                f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). "
            )
            self.best_loss = val_loss
            self.counter = 0
            return True

        print(
            f"Validation loss did not decrease ({self.best_loss:.6f} --> {val_loss:.6f}). "
        )
        self.counter += 1
        if self.counter >= self.patience and self.patience > 0:
            self.early_stop = True
        return False
