# Learning-based schemes
These schemes are implemented using Tensorflow. We support QAM4, QAM16 and QAM64 modulations. Number of transmit and receive anttenas can be set using ``--x-size`` and ``--y-size``, respectively. The range of training and test signal-to-noise ratio (SNR) is set with ``--snr-min`` and ``--snr-max`` arguments. Please use ``--data`` flag in order to specify using specific channels dataset. Not feeding this flag is interpreted for independent and identically distributed Gaussian channels.

## Online training
For online training algorithm run:
```
python3 onlineTraining.py  --x-size 4 --y-size 32 --snr-min 10 --snr-max 15 --layers 10 -lr 1e-3 --batch-size 500 --train-iterations 1000 -mod QAM_16  --test-batch-size 5000 --linear MMNet  --denoiser MMNet --data --channels-dir ../csi_data/H_tensor.npy --output-dir ../results/
```
