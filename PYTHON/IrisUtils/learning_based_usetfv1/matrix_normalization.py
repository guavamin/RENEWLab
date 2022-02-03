import sys
import numpy as np
import h5py
import math

path = '../csi_data/'

# channel_sequence = h5py.File(sys.argv[1], 'r')
#
# name = channel_sequence.filename.strip('.hdf5')
#
# print(name)
#
# print(channel_sequence.filename, ":")
#
# print([key for key in channel_sequence.keys()], "\n")
#
# H_r = channel_sequence.get('H_r')
#
# H_i = channel_sequence.get('H_i')
#
# print("H_r :", H_r)
#
# print("\n")
#
# print("H_i :", H_i)
#
# print("\n")
#
# H_matrix = np.array(H_r[:, :, :, :, :] + 1j*H_i[:, :, :, :, :])
# print(H_matrix.shape)

channel_data = np.load('{}csi_data.npy'.format(path))

print('channel_data.shape:', channel_data.shape)
# print('channel_data:', channel_data)

channel_data_sharp = np.squeeze(channel_data)

print('channel_data_sharp:', channel_data_sharp.shape)

# you can randomly choose or choose the first dimension of the "Pilot Repetition"
# or even have average on this dimension
# here I take first dimension as the data
channel_data_sharp = channel_data_sharp[:, :, 0, :, :]

print('channel_data_sharp:', channel_data_sharp.shape)

# eliminate the pilot subcarrier
'''
data_sc_ind = [ 1,  2,  3,  4,  5,  6,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 20, 22, 23, 24, 25, 26, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48,
                49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63]
'''

final_channel_data = np.delete(channel_data_sharp, [6, 20, 31, 45], 3)

print('final_channel_data.shape:', final_channel_data.shape)

final_channel_data /= np.sqrt(np.sum((final_channel_data*final_channel_data.conjugate()).real, axis=(1, 2, 3), keepdims=True) /
                    (np.prod(final_channel_data.shape[1:4])))

print(np.sum((final_channel_data*final_channel_data.conjugate()).real, axis=(1, 2, 3), keepdims=True) /
                    (np.prod(final_channel_data.shape[1:4])))

print(final_channel_data.shape)

np.save('{}H_tensor.npy'.format(path), final_channel_data)

print('-------finish matrix normalization and successfully save the data-------')

