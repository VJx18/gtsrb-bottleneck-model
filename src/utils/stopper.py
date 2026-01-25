

class EarlyStopper:
    def __init__(self, patience=6, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None

    def early_stop(self, val_loss, model):

        if (val_loss < self.best_score - self.min_delta):
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
            print(f"Validation loss improved to {self.best_loss:.6f}. Saving best model...")
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if (self.counter >= self.patience):
                return True
        return False
    
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)