''' Utils for CLAD and CLOSR loss implementations
'''

import torch as T
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# -- cosine distance function
def cosdist(a, b = None):
    a_norm = F.normalize(a, p=2, dim=-1)
    
    if b is not None:
        b_norm = F.normalize(b, p=2, dim=-1)
    else:
        b_norm = a_norm.clone()
        
    # Compute cosine similarity
    similarity = T.mm(a_norm, b_norm.T)
    
    # Since cosine distance = 1 - cosine similarity
    cosine_distance = (1 - similarity)/2 # divide by 2 gives range [0, 1]
    
    return cosine_distance

def batched_cosdist(x: Tensor) -> Tensor:
    # x: [B, C, D] → [C, B, D]
    x = x.permute(1, 0, 2)
    
    # Normalize along D
    x_norm = F.normalize(x, p=2, dim=-1)  # [C, B, D]

    # Compute cosine similarity using batch matmul:
    # [C, B, D] @ [C, D, B] → [C, B, B]
    sim = T.bmm(x_norm, x_norm.transpose(1, 2))

    # Cosine distance = (1 - sim)/2
    dist = (1 - sim) / 2

    return dist  # [C, B, B]

# -- loss wrappers
class BaseLoss(nn.Module):
    def __init__(self, loss, *args, batch_aug = Identity(), **kwargs):
        super().__init__()
        self.loss = loss(*args, **kwargs)
        self.batch_aug = batch_aug        

    def forward(self, model, x, y, mixed_precision, training):
        logits, y_pred, y_true = self.feed_model(model, x, y, mixed_precision)
        return self.loss(logits,y_pred), self.calc_metric(logits.clone().detach(), y_true)
    
    def get_batch(self, model: nn.Module, x: Tensor, y:Tensor, mixed_precision):
        device = next(model.parameters()).device
        x,y = process_batch((x,y), device, mixed_precision)
        return x,y

    def apply_aug(self, model: nn.Module,  x: Tensor, y: Tensor):
        y_true = y.clone()
        if model.training:
            x, y = self.batch_aug(x, y)
        return x, y, y_true

    def get_model_input(self, model: nn.Module, x: Tensor, y: Tensor, mixed_precision):
        x,y = self.get_batch(model, x, y, mixed_precision)
        x, y, y_true = self.apply_aug(model, x, y)
        return x, y, y_true

    def feed_model(self, model, x, y, mixed_precision):
        x, y, y_true = self.get_model_input(model, x, y, mixed_precision)
        x = model(x)
        return x, y, y_true

    def process_batch(self, x, y):
        return self.batch_aug(x, y)
    
    def calc_metric(self, logits, y_true):
        return T.tensor(0.0 )
    
    def get_epoch_metrics(self, training = True, world_size = 0):
        return {}

class SupervisedLoss(BaseLoss):
    def __init__(
        self, 
        loss, 
        *args, 
        batch_aug = Identity(), 
        cache_labels: bool = True, 
        **kwargs
    ) -> None:

        super().__init__(
            loss, 
            *args, 
            batch_aug = batch_aug,
            **kwargs 
            )
        
        self.cache_labels_ = cache_labels
        self.cached_true_train, self.cached_pred_train = None, None
        self.cached_true_val, self.cached_pred_val = None, None

    def forward(self, model, x, y, mixed_precision, training):
        logits, y_pred, y_true = self.feed_model(model, x, y, mixed_precision)
        if self.cache_labels_:
            self.cache_labels(logits, y_true.to(logits.device), training = model.training)

        return self.loss(logits,y_pred), self.calc_metric(logits.clone().detach(), y_true)

    def calc_metric(self, logits, y_true):
        if logits is None and y_true is None:
            return T.tensor(0.0)
        else:
            return self.calc_acc(logits, y_true)
    
    def calc_acc(self, logits, y_true):
        y_pred = T.argmax(logits, dim = -1)
        correct = (y_pred == y_true).float().sum()
        return correct/y_true.size(0)
    
    def calc_precision(self, logits, y_true):
        y_pred = T.argmax(logits, dim=1)
        true_positives = ((y_pred == 1) & (y_true == 1)).float().sum()
        false_positives = ((y_pred == 1) & (y_true == 0)).float().sum()
        
        if (true_positives + false_positives) == 0:
            return T.tensor(1.0)  # Return 1 if there are no predicted positives to avoid division by zero
        
        precision = true_positives / (true_positives + false_positives)
        return precision
    
    def cache_labels(self, logits, y_true, training):
        if self.cache_labels_:
            y_pred = T.argmax(logits, dim = -1)
            if training:
                if self.cached_true_train is not None:
                    self.cached_true_train = T.cat((self.cached_true_train, y_true))
                    self.cached_pred_train = T.cat((self.cached_pred_train, y_pred))
                else:
                    self.cached_true_train = y_true
                    self.cached_pred_train = y_pred
            else:
                if self.cached_true_val is not None:
                    self.cached_true_val = T.cat((self.cached_true_val, y_true))
                    self.cached_pred_val = T.cat((self.cached_pred_val, y_pred))
                else:
                    self.cached_true_val = y_true
                    self.cached_pred_val = y_pred
                
    #FIXME USE ONE CACHE AND CLEAR AFTER TRAINING
    def clear_cache(self, training):
        if training:
            self.cached_true_train, self.cached_pred_train = None, None
        else:
            self.cached_true_val, self.cached_pred_val = None, None

    def get_epoch_metrics(self, training, world_size = 0):
        metric_dict = None
        if self.cache_labels_:
            if training:
                y_true = gather_concat(self.cached_true_train, world_size).cpu().detach().numpy()
                y_pred = gather_concat(self.cached_pred_train, world_size).cpu().detach().numpy()
            
                metric_dict = model_eval(y_true, y_pred, label='train')
                self.clear_cache(True)
            else:
                y_true = gather_concat(self.cached_true_val, world_size)
                y_pred = gather_concat(self.cached_pred_val, world_size)
                
                if y_true is not None: 
                    y_true = y_true.cpu().detach().numpy()
                    y_pred = y_pred.cpu().detach().numpy()
                    metric_dict = model_eval(y_true, y_pred, label='val')
                    self.clear_cache(False)

        return metric_dict if metric_dict is not None else {}
    


class TripletLossWrapper(SupervisedLoss):
    def __init__(self, loss, *args, online_probe: bool = True, batch_aug = Identity(), **kwargs):
       self.online_probe = online_probe
       super().__init__(* args, loss = loss, batch_aug = batch_aug, **kwargs)

    def calc_metric(self, logits, y_true):
        return 1 - self.loss.get_fraction_pos()

    def feed_model(self, model, x, y, mixed_precision):
        x, y, y_true = self.get_model_input(model, x, y, mixed_precision)
        x = model.forward_features(x)
        return x, y, y_true

    def forward(self, model, x, y, mixed_precision, training):
        logits, y, y_true = self.feed_model(model, x, y, mixed_precision)
        loss = self.loss(model.proj(logits), y)
        
        if self.online_probe:
            logits_ce = logits.clone().detach()
            logits_ce.requires_grad_(True)
            
            logits_ce = model.probe(logits_ce)
            ce_loss = F.cross_entropy(logits_ce, y) 
            loss = (loss, ce_loss)

            if self.cache_labels_:
                self.cache_labels(logits_ce, y_true.to(logits.device), training = model.training)

        return loss, self.calc_metric(logits.clone().detach(), y_true)
