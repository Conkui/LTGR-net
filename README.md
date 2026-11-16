# LTGR-net--Spine reconstruction

This repository contains our preprocessing and paired MRI display tools.
Model and training code will follow after publication.

```text
configs/
  data/
  models/
scripts/
  data/
  models/
  training/
  evaluation/
  figures/
tests/
```

Requires Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python -m scripts.data.prepare_spider --data-root path/to/spider --series SERIES_NAME
python -m scripts.figures.spider_ssl_paired_mri_demo --data-root path/to/spider --series SERIES_NAME
```

Use the same series name for `images/*.mha` and `masks/*.mha`.
Data are obtained separately and are not distributed here.

```bash
python -B -m unittest discover -s tests
```
