# Instruction
These schemes are implemented using Tensorflow. We support QAM4, QAM16 and QAM64 modulations. Number of transmit and receive anttenas can be set using ``--x-size`` and ``--y-size``, respectively. The range of training and test signal-to-noise ratio (SNR) is set with ``--snr-min`` and ``--snr-max`` arguments. Please use ``--data`` flag in order to specify using specific channels dataset. Not feeding this flag is interpreted for independent and identically distributed Gaussian channels.

In this section, I mainly try to fit the real-world scenrio data into the MMNet Model to see how it works.

## Step
I firstly normalize the csi_data(channel state information) with the code `normalization.py` to make sure the data satisfy the formula in the MMNet paper.

Secondly, I train the model by using first half of the data of 0 subcarrier.

Then, I test the effectiveness of the model by using rest half of the data of 0 subcarrier to see whether the temporal correlation work in the real-world environment.

Last but not least, I test the effectiveness of the model by using the data of other subcarriers to see whether the spectral correlation work in the real-world environment.

## Online training
For online training algorithm run:
```
python3 onlineTraining.py  --x-size 4 --y-size 32 --snr-min 10 --snr-max 15 --layers 10 -lr 1e-3 --batch-size 500 --train-iterations 1000 -mod QAM_16  --test-batch-size 5000 --linear MMNet  --denoiser MMNet --data --channels-dir ../csi_data/H_tensor.npy --output-dir ../results/
```
