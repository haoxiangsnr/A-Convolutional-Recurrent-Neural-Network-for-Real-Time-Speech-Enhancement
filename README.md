# A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement

Implement [A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement](https://arxiv.org/abs/1805.00579) by PyTorch.

## Performance

Model performance on private dataset, for reference only

| Experiment |  | 5dB |  |  | 10dB |  | Average | Comment |
| :---: | :---: | :---: | :---: | --- | --- | --- | --- | --- |
|  | 0.5m | 1m | 2m | 0.5m | 1m | 2m | | Distance to the microphone |
| Mixture | 2.33 | 2.17 | 1.85 | 2.44 | 2.27 | 1.94 | 2.167 |  |
| LSTM | 2.62 | 2.49 | 2.02 | 2.71 | 2.53 | 2.13 | 2.417 |  |
| Our implementation |2.630 | 2.458 | 2.086 | 2.729 | 2.527 | 2.172 | 2.434 |  |
| Our implementation (LN) | 2.703 | 2.461 | 1.961 | 2.796 | 2.548 | 2.181 | 2.442 | Replace all batch norm with layer norm |

## Dependencies

- Python3
- torch==1.1.0
- librosa==0.7.0
- SoundFile==0.10.2
- tensorboard==1.14.0
- tensorboard==1.13.1(for visualization only)
- pypesq==1.0, `pip install https://github.com/vBaiCai/python-pesq/archive/master.zip`
- pystoi==0.2.2
- matplotlib==3.1.0
- tqdm==4.32.2

## Reference

- [CRNN_mapping_baseline](https://github.com/YangYang/CRNN_mapping_baseline)
- [A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang1.interspeech18.pdf)
- [EHNet](https://github.com/ododoyo/EHNet)
- [Convolutional-Recurrent Neural Networks for Speech Enhancement](https://arxiv.org/abs/1805.00579)
