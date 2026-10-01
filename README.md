# panns-infer

PANNs CNN14 AudioSet tagging and native audio embeddings, as an independent inference-only package.

## Why this exists

[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) introduced reusable
pretrained audio neural networks. Its research repository includes training,
evaluation and historically pinned dependencies. This package extracts CNN14
inference for modern Python and Torch while preserving its checkpoint and
native outputs.

## Acknowledgments

**All model credit belongs to the original PANNs authors. Please support and cite their work.**

- [Qiuqiang Kong and the original PANNs repository](https://github.com/qiuqiangkong/audioset_tagging_cnn).
- Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang and Mark D. Plumbley,
  verified against the [original paper](https://arxiv.org/abs/1912.10211).
- [Zenodo record 3987831](https://zenodo.org/records/3987831), the original pretrained checkpoint host.
- [AudioSet](https://research.google.com/audioset/), its 527-class ontology and label mapping.

## Citation

@misc{kong2020pannslargescalepretrainedaudio,
      title={PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition}, 
      author={Qiuqiang Kong and Yin Cao and Turab Iqbal and Yuxuan Wang and Wenwu Wang and Mark D. Plumbley},
      year={2020},
      eprint={1912.10211},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/1912.10211}, 
}