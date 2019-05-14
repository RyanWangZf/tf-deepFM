# -*- coding: utf-8 -*-
class Config:

    hidden_units = [400, 400, 400] # deep nn's layers
    
    k = 32 # latent dim
    n = 999997 # num of feature
    m = 39 # num of field

    num_epoch = 20
    l2_norm = 1e-6 # l2 norm imposed on weights
    learning_rate = 1e-3
    batch_size = 5000
    shuffle = True

    use_gpu = False
    model_dir = "./ckpt"

config = Config()