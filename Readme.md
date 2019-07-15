# Deep Learning. Part Two.

### Goal 
Main goal of this project is to replace standard Pytorch implementation by custom methods on matrix level in both scalar in vector form.

### Network structure
Implemented network conists from: 
1) Conv layer 
2) Maxpool layer, 
3) Reshape layer 
4) FC layer with Relu
5) Plain FC layer
6) Softmax activation

### How to use this repo
Simply run `python simple_conv_net_train.py`
If you want to check if custom layers are similiar to pytorch run `python simple_conv_net_train.py --check_consistency`
Also, you can specify implementation version as follows `python simple_conv_net_train.py --version self-vector` (for scalar go with `--version self-scalar` and for pytorch with `--version torch`)

### Reported time & accuracy
I performed experiments on four Titan X with batch size equals to 1024 (just to not wait too long).

1) Pytorch implementation takes on average 13.83 seconds to train one epoch and I've got 96.36% accuracy after 15 epoch.
2) Custom vector implementation takes on average 13.21 seconds to train one epoch and I've got 96.37% accuracy after 15 epoch.
3) Custom scalar implementation takes just a ton of time... it's still running...
