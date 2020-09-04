# A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement

A minimum unofficial implementation of the [A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement (CRN)](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1405.pdf) using PyTorch.

## ToDo
- [x] Real-time version
- [x] Update trainer
- [x] Visualization of the spectrogram and the metrics (PESQ, STOI, SI-SDR) in the training
- [ ] More docs

## Usage

Training:

```
TODO
```

Inference:

``
python inference.py \
    -C config/inference/basic.json5 \
    -cp ~/Experiments/CRN/baseline_model/checkpoints/latest_model.tar \
     -dist ./enhanced
``

## Performance

PESQ metric on a private dataset, for reference only

| Experiment |  | 5dB |  |  | 10dB |  | Average | Comment |
| :---: | :---: | :---: | :---: | --- | --- | --- | --- | --- |
|  | 0.5m | 1m | 2m | 0.5m | 1m | 2m | | Distance to microphone |
| Mixture | 2.33 | 2.17 | 1.85 | 2.44 | 2.27 | 1.94 | 2.167 |  |
| LSTM | 2.62 | 2.49 | 2.02 | 2.71 | 2.53 | 2.13 | 2.417 |  |
| Our implementation |2.630 | 2.458 | 2.086 | 2.729 | 2.527 | 2.172 | 2.434 |  |
| Our implementation (LN) | 2.703 | 2.461 | 1.961 | 2.796 | 2.548 | 2.181 | 2.442 | Replace all batch norm with layer norm |


PESQ, STOI, SI-SDR metrics on SEGAN dataset.

| Experiment | PESQ | SI-SDR | STOI |
| --- | --- | --- | --- |
|crn | 2.528| 17.71| 0.9325|
|crn_mask |2.606 |17.84 |0.9382|

## Dependencies

- Python==3.\*.\*
- torch==1.\*
- librosa==0.7.0
- tensorboard
- pesq
- pystoi
- matplotlib
- tqdm

## References

- [CRNN_mapping_baseline](https://github.com/YangYang/CRNN_mapping_baseline)
- [A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang1.interspeech18.pdf)
- [EHNet](https://github.com/ododoyo/EHNet)
- [Convolutional-Recurrent Neural Networks for Speech Enhancement](https://arxiv.org/abs/1805.00579)
