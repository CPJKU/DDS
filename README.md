# DDS

This repository hosts the implementation of Differentiable Dictionary Search (DDS) method.

## Install

In your Python (tested with 3.8) environment of choice:

`pip install -r requirements.txt`

Link the MAPS dataset:

```bash
path_to_maps=/data/MAPS # or wherever your unpacked MAPS resides
mkdir ./data
ln -s $path_to_maps ./data/MAPS
```

## Use

* [train_range.py](train_range.py) trains a set of single-source models for specified note range
* [train_ms.py](train_ms.py) trains a single multi-source model for specified note range
* [test_RAND.py](test_RAND.py) evaluates specified method on specified subset of MAPS/RAND
* [test_MUS.py](test_MUS.py) evaluates specified method on specified piece from MAPS/MUS

More detailed instructions with examples in curation.

## License

All software provided in this repository is subject to the [CRAPL license](CRAPL-LICENSE.txt).
