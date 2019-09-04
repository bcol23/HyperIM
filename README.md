# HyperIM

PyTorch implementation of [Hyperbolic Interaction Model For Hierarchical Multi-Label Classification](https://arxiv.org/abs/1905.10802)

## Requirements

- torch>=1.0.0
- geoopt (`$ pip install git+https://github.com/geoopt/geoopt.git`)
- numpy
- scipy
- pandas
- tqdm
## Instruction

Run HyperIM via

```
$ python HyperIM.py
```

or run EuclideanIM via

```
$ python EuclideanIM.py
```

Alternatively use the two Jupyter notebooks.

## Data

*X_train* and *X_test* should be dense *numpy* array with shape (instance_num, word_num), *y_train* and *y_test* should be one-hot sparse *scipy* array with shape (instance_num, label_num). Sample data is provided in `./data/sample/`.

The multi-label text classification datasets equipped with hierarchically structured labels ([*RCV1*](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm), [*Zhihu*](https://biendata.com/competition/zhihu/) and [*WikiLSHTC*](http://lshtc.iit.demokritos.gr/)) are publicly available.

## Pre-trained embeddings

Hyperbolic word embeddings can be trained following [Poincaré GloVe](https://github.com/alex-tifrea/poincare_glove). Pre-trained word embeddings should have shape (vocab_size, embed_dim).

The label hierarchy can be embedded using the [gensim](https://radimrehurek.com/gensim/) implementation of the [Poincaré embeddings](https://github.com/facebookresearch/poincare-embeddings), specified in [Train and use Poincaré embeddings](https://radimrehurek.com/gensim/models/poincare.html). Label embeddings should have shape (label_num, embed_dim). Note that the index of labels in the label embeddings should be consistent with *y_train* and *y_test*.

Use them accordingly in `HyperIM.py` and `EuclideanIM.py`.

## Citation
If you find this code useful for your research, please cite the following paper in your publication:

```
@article{chen2019hyperbolic,
  title={Hyperbolic Interaction Model For Hierarchical Multi-Label Classification},
  author={Chen, Boli and Huang, Xin and Xiao, Lin and Cai, Zixin and Jing, Liping },
  journal={arXiv preprint arXiv:1905.10802},
  year={2019}
}
```
