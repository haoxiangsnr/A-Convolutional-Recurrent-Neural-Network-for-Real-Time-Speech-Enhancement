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
python train.py -C config/train/baseline_model.json5
```

Inference:

```
python inference.py \
    -C config/inference/basic.json5 \
    -cp ~/Experiments/CRN/baseline_model/checkpoints/latest_model.tar \
    -dist ./enhanced
```

Check out the README of [Wave-U-Net for SE](https://github.com/haoxiangsnr/Wave-U-Net-for-Speech-Enhancement) to learn more.

## Performance

PESQ, STOI, SI-SDR on DEMAND - Voice Bank test dataset, for reference only:

| Experiment | PESQ | SI-SDR | STOI |
| --- | --- | --- | --- |
|Noisy | 1.979 | 8.511| 0.9258|
|CRN | 2.528| 17.71| 0.9325|
|CRN signal approximation  |2.606 |17.84 |0.9382|

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
