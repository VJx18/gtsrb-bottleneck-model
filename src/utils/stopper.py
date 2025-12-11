

class EarlyStopper:
    def __init__(self, patience=10, min_delta=10):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = float("inf")

    def early_stop(self, val_loss):
        if (val_loss < self.min_val_loss):
            self.min_val_loss = val_loss
        elif (val_loss > self.min_val_loss + self.min_delta):
            self.counter += 1
            if (self.counter >= self.patience):
                print(f"Early stop! \n")
                return True
        return False