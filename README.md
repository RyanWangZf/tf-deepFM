# tf-deepFM
A deep factorization machine implemented via Tensorflow.

```python
from config import config
from dataset import Dataset
from deepFM import deepFM
deepfm = deepFM(config)
tr_filename = "data/criteo.tr.r100.gbdt0.ffm"
va_filename = "data/criteo.va.r100.gbdt0.ffm"
dataset =\
   Dataset(tr_filename,va_filename,config.batch_size,config.shuffle)
deepfm.train(dataset)
```

