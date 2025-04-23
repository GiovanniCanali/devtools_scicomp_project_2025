class EarlyStopping:
    """
    EarlyStopping class to stop training when validation loss does not improve.
    This class monitors the validation loss and stops training if it does not
    improve for a specified number of steps (patience).
    """

    def __init__(self, patience):
        """ """
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        if patience == 0:
            self.__call__ = lambda *args, **kwargs: None

    def __call__(self, val_loss):
        if self.patience != 0:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
