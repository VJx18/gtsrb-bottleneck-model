

class EarlyStopper:
    def __init__(self, patience=6, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_model_state = None
        self.best_score = None

    def early_stop(self, val_loss, model):
        
        score = -val_loss
        print('min_delta:', self.min_delta)

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif (score > self.best_score + self.min_delta):
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
            print(f"Validation loss improved to {-self.best_score:.6f}. Saving best model...")      
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if (self.counter >= self.patience):
                return True
        return False
    
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)