#!/usr/bin/python3
"""
 hdf5_lib.py

 Library handling recorded hdf5 file from channel sounding (see Sounder/).

 Author(s):
             C. Nicolas Barati: nicobarati@rice.edu
             Oscar Bejarano: obejarano@rice.edu
             Rahman Doost-Mohammady: doost@rice.edu

---------------------------------------------------------------------
 Copyright © 2018-2019. Rice University.
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
---------------------------------------------------------------------
"""

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import collections
import time
from optparse import OptionParser
from channel_analysis import *
import multiprocessing as mp
import extract_pilots_data as epd

#@staticmethod
def csi_from_pilots(pilots_dump, z_padding=150, fft_size=64, cp=16, frm_st_idx=0, frame_to_plot=0, ref_ant=0, ref_user = 0):
    """
    Finds the end of the pilots' frames, finds all the lts indices relative to that.
    Divides the data with lts sequences, calculates csi per lts, csi per frame, csi total.
    """
    print("********************* csi_from_pilots(): *********************")

    # Reviewing options and vars:
    show_plot = True
    debug = False
    test_mf = False
    write_to_file = True
    legacy = False

    # dimensions of pilots_dump
    n_frame = pilots_dump.shape[0]      # no. of captured frames
    n_cell = pilots_dump.shape[1]       # no. of cells
    n_ue = pilots_dump.shape[2]         # no. of UEs
    n_ant = pilots_dump.shape[3]        # no. of BS antennas
    n_iq = pilots_dump.shape[4]         # no. of IQ samples per frame

    if debug:
        print("input : z_padding = {}, fft_size={}, cp={}, frm_st_idx = {}, frame_to_plot = {}, ref_ant={}, ref_user = {}".format(
            z_padding, fft_size, cp, frm_st_idx, frame_to_plot, ref_ant, ref_user))
        print("n_frame = {}, n_cell = {}, n_ue = {}, n_ant = {}, n_iq = {}".format(
            n_frame, n_cell, n_ue, n_ant, n_iq))

    if ((n_iq % 2) != 0):
        print("Size of iq samples:".format(n_iq))
        raise Exception(
            ' **** The length of iq samples per frames HAS to be an even number! **** ')

    n_cmpx = n_iq // 2  # no. of complex samples
    # no. of complex samples in a P subframe without pre- and post- fixes
    n_csamp = n_cmpx - z_padding
    if legacy:
        # even indices: real part of iq      --> ATTENTION: I and Q are flipped at RX for some weird reason! So, even starts from 1!
        idx_e = np.arange(1, n_iq, 2)
        # odd  indices: imaginary part of iq --> ATTENTION: I and Q are flipped at RX for some weird reason! So, odd starts from 0!
        idx_o = np.arange(0, n_iq, 2)
    else:
        idx_e = np.arange(0, n_iq, 2)       # even indices: real part of iq
        # odd  indices: imaginary part of iq
        idx_o = np.arange(1, n_iq, 2)

    # make a new data structure where the iq samples become complex numbers
    cmpx_pilots = (pilots_dump[:, :, :, :, idx_e] +
                   1j*pilots_dump[:, :, :, :, idx_o])*2**-15

    # take a time-domain lts sequence, concatenate more copies, flip, conjugate
    lts_t, lts_f = generate_training_seq(preamble_type='lts', seq_length=[
    ], cp = cp, upsample=1, reps=[])    # TD LTS sequences (x2.5), FD LTS sequences
    # DON'T assume cp!
    lts_tmp = lts_t[- cp - fft_size:]
    n_lts = len(lts_tmp)
    # no. of LTS sequences in a pilot SF
    k_lts = n_csamp // n_lts
    # concatenate k LTS's to filter/correlate below
    lts_seq = np.tile(lts_tmp, k_lts)
    lts_seq = lts_seq[::-1]                         # flip
    # conjugate the local LTS sequence
    lts_seq_conj = np.conjugate(lts_seq)
    # length of the local LTS seq.
    l_lts_fc = len(lts_seq_conj)

    if debug:
        print("cmpx_pilots.shape = {}, lts_t.shape = {}".format(
            cmpx_pilots.shape, lts_t.shape))
        #print("idx_e= {}, idx_o= {}".format(idx_e, idx_o))
        print("n_cmpx = {}, n_csamp = {}, n_lts = {}, k_lts = {}, lts_seq_conj.shape = {}".format(
            n_cmpx, n_csamp, n_lts, k_lts, lts_seq_conj.shape))

    # debug/ testing
    if debug:
        z_pre = np.zeros(82, dtype='complex64')
        z_post = np.zeros(68, dtype='complex64')
        lts_t_rep = np.tile(lts_tmp, k_lts)
        lts_t_rep_tst = np.append(z_pre, lts_t_rep)
        lts_t_rep_tst = np.append(lts_t_rep_tst, z_post)

        if test_mf:
            w = np.random.normal(0, 0.1/2, len(lts_t_rep_tst)) + \
                1j*np.random.normal(0, 0.1/2, len(lts_t_rep_tst))
            lts_t_rep_tst = lts_t_rep_tst + w
            cmpx_pilots = np.tile(
                lts_t_rep_tst, (n_frame, cmpx_pilots.shape[1], cmpx_pilots.shape[2], cmpx_pilots.shape[3], 1))
            print("if test_mf: Shape of lts_t_rep_tst: {} , cmpx_pilots.shape = {}".format(
                lts_t_rep_tst.shape, cmpx_pilots.shape))

    ## normalized matched filter
    #a = 1
    #unos = np.ones(l_lts_fc)
    #v0 = signal.lfilter(lts_seq_conj, a, cmpx_pilots, axis=4)
    #v1 = signal.lfilter(unos, a, (abs(cmpx_pilots)**2), axis=4)
    #m_filt = (np.abs(v0)**2)/v1

    ## clean up nan samples: replace nan with -1
    #nan_indices = np.argwhere(np.isnan(m_filt))
    #m_filt[np.isnan(m_filt)] = -0.5  # the only negative value in m_filt

    #if write_to_file:
    #    # write the nan_indices into a file
    #    np.savetxt("nan_indices.txt", nan_indices, fmt='%i')

    #if debug:
    #    print("Shape of truncated complex pilots: {} , l_lts_fc = {}, v0.shape = {}, v1.shape = {}, m_filt.shape = {}".
    #          format(cmpx_pilots.shape, l_lts_fc, v0.shape, v1.shape, m_filt.shape))

    #rho_max = np.amax(m_filt, axis=4)         # maximum peak per SF per antenna
    #rho_min = np.amin(m_filt, axis=4)        # minimum peak per SF per antenna
    #ipos = np.argmax(m_filt, axis=4)          # positons of the max peaks
    #sf_start = ipos - l_lts_fc + 1             # start of every received SF
    ## get rid of negative indices in case of an incorrect peak
    #sf_start = np.where(sf_start < 0, 0, sf_start)

    ## get the pilot samples from the cmpx_pilots array and reshape for k_lts LTS pilots:
    #pilots_rx_t = np.empty(
    #    [n_frame, n_cell, n_ue, n_ant, k_lts * n_lts], dtype='complex64')

    #indexing_start = time.time()
    #for i in range(n_frame):
    #    for j in range(n_cell):
    #        for k in range(n_ue):
    #            for l in range(n_ant):
    #                pilots_rx_t[i, j, k, l, :] = cmpx_pilots[i, j, k, l,
    #                                                         sf_start[i, j, k, l]:  sf_start[i, j, k, l] + (k_lts * n_lts)]

    #indexing_end = time.time()

    pool = mp.Pool(mp.cpu_count())
    m_filt = np.empty(
        [n_frame, n_cell, n_ue, n_ant, n_cmpx], dtype='complex64')
    pilots_rx_t = np.empty(
        [n_frame, n_cell, n_ue, n_ant, k_lts * n_lts], dtype='complex64')
    sf_start = np.empty(
        [n_frame, n_cell, n_ue, n_ant], dtype='int')
    indexing_start = time.time()
    for j in range(n_cell):
        result_objects = [pool.apply_async(epd.extract_pilots_data, args=(cmpx_pilots[i, j, :, :, :], lts_seq_conj, k_lts, n_lts, i)) for i in range(n_frame)]
        for i in range(n_frame):
            pilots_rx_t[i, j, :, :, :] = result_objects[i].get()[1]
            m_filt[i, j, :, :, :] = result_objects[i].get()[2]
            sf_start[i, j, :, :] = result_objects[i].get()[3]
    indexing_end = time.time()
    pool.close()

    # WZC: added pilots_rx_t_btc (pilots_rx_t before truncation) for output
    pilots_rx_t_btc = np.copy(pilots_rx_t)

    # *************** This fancy indexing is slower than the for loop! **************
#    aaa= np.reshape(cmpx_pilots, (n_frame*n_cell* n_ue * n_ant, n_cmpx))
#    idxx = np.expand_dims(sf_start.flatten(), axis=1)
#    idxx = np.tile(idxx, (1,k_lts*n_lts))
#    idxx = idxx + np.arange(k_lts*n_lts)
#    indexing_start2 = time.time()
#    m,n = aaa.shape
#    #bb = aaa[np.arange(aaa.shape[0])[:,None],idxx]
#    bb = np.take(aaa,idxx + n*np.arange(m)[:,None])
#    indexing_end2 = time.time()
#    cc = np.reshape(bb,(n_frame,n_cell, n_ue , n_ant, k_lts * n_lts) )
#    if debug:
#       print("Shape of: aaa  = {}, bb: {}, cc: {}, flattened sf_start: {}\n".format(aaa.shape, bb.shape, cc.shape, sf_start.flatten().shape))
#       print("Indexing time 2: %f \n" % ( indexing_end2 -indexing_start2) )

    if debug:
        print("Shape of: pilots_rx_t before truncation: {}\n".format(
            pilots_rx_t.shape))

    pilots_rx_t = np.reshape(
        pilots_rx_t, (n_frame, n_cell, n_ue, n_ant, k_lts, n_lts))
    pilots_rx_t = np.delete(pilots_rx_t, range(fft_size, n_lts), 5)

    if debug:
        print("Indexing time: %f \n" % (indexing_end - indexing_start))
        print("Shape of: pilots_rx_t = {}\n".format(pilots_rx_t.shape))

        # WZC: rho_max might not be defined in current version
        # print("Shape of: rho_max = {}, rho_min = {}, ipos = {}, sf_start = {}".format(
        #     rho_max.shape, rho_min.shape, ipos.shape, sf_start.shape))

    # take fft and get the raw CSI matrix (no averaging)
    # align SCs based on how they were Tx-ec
    lts_f_shft = np.fft.fftshift(lts_f)
    pilots_rx_f = np.fft.fft(pilots_rx_t, fft_size, 5)      # take FFT
    # find the zero SCs corresponding to lts_f_shft
    zero_sc = np.where(lts_f_shft == 0)[0]
    # remove zero subcarriers
    lts_f_nzsc = np.delete(lts_f_shft, zero_sc)
    # remove zero subcarriers
    pilots_rx_f = np.delete(pilots_rx_f, zero_sc, 5)
    # take channel estimate by dividing with the non-zero elements of lts_f_shft
    csi = pilots_rx_f / lts_f_nzsc
    # unecessary step: just to make it in accordance to lts_f as returned by generate_training_seq()
    csi = np.fft.fftshift(csi, 5)

    if debug:
        # # WZC: rho_max might not be defined in current version
        # print(">>>> number of NaN indices = {} NaN indices =\n{}".format(
        #     nan_indices.shape, nan_indices))
        print("Shape of: csi = {}\n".format(csi.shape))

    # plot something to see if it worked!
    if show_plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.grid(True)
        ax1.set_title(
            'channel_analysis:csi_from_pilots(): Re of Rx pilot - ref frame {} ref ant {} ref user {}'.format(frame_to_plot, ref_ant, ref_user))
        if debug:
            print("cmpx_pilots.shape = {}".format(cmpx_pilots.shape))

        ax1.plot(
            np.real(cmpx_pilots[frame_to_plot - frm_st_idx, 0, ref_user, ref_ant, :]))

        if debug:
            loc_sec = lts_t_rep_tst
        else:
            z_pre = np.zeros(82, dtype='complex64')
            z_post = np.zeros(68, dtype='complex64')
            lts_t_rep = np.tile(lts_tmp, k_lts)
            loc_sec = np.append(z_pre, lts_t_rep)
            loc_sec = np.append(loc_sec, z_post)
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.grid(True)
        ax2.set_title(
            'channel_analysis:csi_from_pilots(): Local LTS sequence zero padded')
        ax2.plot(loc_sec)

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.grid(True)
        ax3.set_title(
            'channel_analysis:csi_from_pilots(): MF (uncleared peaks) - ref frame {} ref ant {} ref user {}'.format(frame_to_plot, ref_ant, ref_user))
        ax3.stem(m_filt[frame_to_plot - frm_st_idx, 0, ref_user, ref_ant, :])
        ax3.set_xlabel('Samples')
        # plt.show()

    print("********************* ******************** *********************\n")
    return csi, m_filt, sf_start, cmpx_pilots, k_lts, n_lts, sf_start, pilots_rx_t_btc
    # add frame_start for plot indexing!
    # WZC: add pilots_rx_t_btc

class hdf5_lib:

    def __init__(self, filename, n_frames_to_inspect=0, n_fr_insp_st = 0, sub_sample = 0):
        self.h5file = None
        self.filename = filename
        self.dirpath = '/'.join(filename.split('/')[:-1])
        self.h5struct = []
        self.data = []
        self.metadata = {}
        self.pilot_samples = []
        self.uplink_samples = []
        self.noise_samples = []
        self.n_frm_st = n_fr_insp_st                                # index of last frame
        self.n_frm_end = self.n_frm_st + n_frames_to_inspect    # index of last frame in the range of n_frames_to_inspect
        self.sub_sample = sub_sample
        self.open_hdf5()
        self.get_data()
        self.get_metadata()

    def open_hdf5(self):
        """
        Get the most recent log file, open it if necessary.
        """
        if (not self.h5file) or (self.filename != self.h5file.filename):
            # if it's closed, e.g. for the C version, open it
            print('Opening %s...' % self.filename)
            try:
                self.h5file = h5py.File(self.filename, 'r')
            except OSError:
                print("File not found. Terminating program now")
                sys.exit(0)
        # return self.h5file

    def get_data(self):
        """
        Parse file to retrieve metadata and data.
        HDF5 file has been written in DataRecorder.cpp (in Sounder folder)

        Output:
            Data (hierarchy):
                -Path
                -Pilot_Samples
                    --Samples
                -UplinkData
                    --Samples
        Dimensions of input sample data (as shown in DataRecorder.cpp in Sounder):
            - Pilots
                dims_pilot[0] = maxFrame
                dims_pilot[1] = number of cells
                dims_pilot[2] = number of UEs
                dims_pilot[3] = number of antennas (at BS)
                dims_pilot[4] = samples per symbol * 2 (IQ)

            - Uplink Data
                dims_data[0] = maxFrame
                dims_data[1] = number of cells
                dims_data[2] = uplink symbols per frame
                dims_data[3] = number of antennas (at BS)
                dims_data[4] = samples per symbol * 2 (IQ)
        """

        self.data = self.h5file['Data']

        if 'Pilot_Samples' in self.data:
            if self.n_frm_st == self.n_frm_end:
                # Consider the entire dataset (for demos etc)
                self.pilot_samples = self.data['Pilot_Samples']
            else:
                self.pilot_samples = self.data['Pilot_Samples'][self.n_frm_st:self.n_frm_end:self.sub_sample, ...]

        if len(self.data.keys()) > 1:
            print("looking into UplinkData")
            if 'UplinkData' in self.data:
                if self.n_frm_st == self.n_frm_end:
                    # Consider the entire dataset (for demos etc)
                    self.uplink_samples = self.data['UplinkData']
                else:
                    self.uplink_samples = self.data['UplinkData'][self.n_frm_st:self.n_frm_end:self.sub_sample, ...]

            print("looking into Noise Samples (if enabled in Sounder)")
            if 'Noise_Samples' in self.data:
                if self.n_frm_st == self.n_frm_end:
                    # Consider the entire dataset (for demos etc)
                    self.noise_samples = self.data['Noise_Samples']
                else:
                    self.noise_samples = self.data['Noise_Samples'][self.n_frm_st:self.n_frm_end:self.sub_sample, ...]

        return self.data

    def get_metadata(self):
        """
                -Attributes
                        {FREQ, RATE, PREFIX_LEN, POSTFIX_LEN, SLOT_SAMP_LEN, FFT_SIZE, CP_LEN,
                        BEACON_SEQ_TYPE, PILOT_SEQ_TYPE, BS_HUB_ID, BS_SDR_NUM_PER_CELL, BS_SDR_ID, BS_NUM_CELLS,
                        BS_CH_PER_RADIO, BS_FRAME_SCHED, BS_RX_GAIN_A, BS_TX_GAIN_A, BS_RX_GAIN_B, BS_TX_GAIN_B,
                        BS_BEAMSWEEP, BS_BEACON_ANT, BS_NUM_ANT, BS_FRAME_LEN, CL_NUM, CL_CH_PER_RADIO, CL_AGC_EN,
                        CL_RX_GAIN_A, CL_TX_GAIN_A, CL_RX_GAIN_B, CL_TX_GAIN_B, CL_FRAME_SCHED, CL_SDR_ID,
                        CL_MODULATION, UL_SLOTS}
        """

        # Retrieve attributes, translate into python dictionary
        #data = self.data
        self.metadata = dict(self.h5file['Data'].attrs)
        if "CL_SDR_ID" in self.metadata.keys():
            cl_present = True
        else:
            cl_present = False
            print('Client information not present. It is likely the client was run separately')

        bs_id = self.metadata['BS_SDR_ID'].astype(str)
        if bs_id.size == 0:
            raise Exception('Base Station information not present')

        # Data cleanup
        # In OFDM_DATA_CLx and OFDM_PILOT, we have stored both real and imaginary in same vector
        # (i.e., RE1,IM1,RE2,IM2...REm,IM,)

        # Freq-domain Pilot
        if "OFDM_PILOT_F" in self.metadata.keys():
            pilot_vec = self.metadata['OFDM_PILOT_F']
            # some_list[start:stop:step]
            I = pilot_vec[0::2]
            Q = pilot_vec[1::2]
            pilot_complex = I + Q * 1j
            self.metadata['OFDM_PILOT_F'] = pilot_complex

        # Time-domain Pilot
        pilot_vec = self.metadata['OFDM_PILOT']
        # some_list[start:stop:step]
        I = pilot_vec[0::2]
        Q = pilot_vec[1::2]
        pilot_complex = I + Q * 1j
        self.metadata['OFDM_PILOT'] = pilot_complex

        if 'UL_SYMS' in self.metadata:
            n_ul_slots = self.metadata['UL_SYMS']
        elif 'UL_SLOTS' in self.metadata:
            n_ul_slots = self.metadata['UL_SLOTS']
        if n_ul_slots > 0:
            # Phase-Tracking Pilot Subcarrier
            pilot_sc_vals = self.metadata['OFDM_PILOT_SC_VALS']
            pilot_sc_vals_complex = pilot_sc_vals[0::2] + 1j * pilot_sc_vals[1::2]
            self.metadata['OFDM_PILOT_SC_VALS'] = pilot_sc_vals_complex

            # Time-domain OFDM data
            num_cl = np.squeeze(self.metadata['CL_NUM'])
            for clIdx in range(num_cl):
                this_str = 'OFDM_DATA_TIME_CL' + str(clIdx)
                if not this_str in self.metadata.keys():
                    continue
                data_per_cl = np.squeeze(self.metadata[this_str])
                # some_list[start:stop:step]
                if np.any(data_per_cl):
                    # If data present
                    I = np.double(data_per_cl[0::2])
                    Q = np.double(data_per_cl[1::2])
                    IQ = I + Q * 1j
                    self.metadata[this_str] = IQ

            # Frequency-domain OFDM data
            for clIdx in range(num_cl):
                this_str = 'OFDM_DATA_CL' + str(clIdx)
                if not this_str in self.metadata.keys():
                    continue
                data_per_cl = np.squeeze(self.metadata[this_str])
                # some_list[start:stop:step]
                if np.any(data_per_cl):
                    # If data present
                    I = np.double(data_per_cl[0::2])
                    Q = np.double(data_per_cl[1::2])
                    IQ = I + Q * 1j
                    self.metadata[this_str] = IQ
                # print('IQ.shape:', IQ.shape) # (640, )

        return self.metadata

    def filter_pilots(cmpx_pilots, z_padding=150, fft_size=64, cp=16, pilot_type='lts',seq_length=[]):
        """
        """
        # dimensions of cmpx_pilots
        n_frame = cmpx_pilots.shape[0]  # no. of captured frames
        n_cell = cmpx_pilots.shape[1]  # no. of cells
        n_ue = cmpx_pilots.shape[2]  # no. of UEs
        n_ant = cmpx_pilots.shape[3]  # no. of BS antennas
        n_cmpx = cmpx_pilots.shape[4]  # no. of IQ samples per frame

        # no. of complex samples in a P subframe without pre- and post- fixes
        n_csamp = n_cmpx - z_padding

        # take a time-domain lts sequence, concatenate more copies, flip, conjugate
        seq_t, seq_f = generate_training_seq(preamble_type=pilot_type, seq_length=
        seq_length, cp=cp, upsample=1, reps=[])  # TD LTS sequences (x2.5), FD LTS sequences
        # DON'T assume cp!
        seq_tmp = seq_t[- cp - fft_size:]
        seq_len = len(seq_tmp)
        # no. of LTS sequences in a pilot SF
        seq_num = n_csamp // seq_len
        # concatenate k LTS's to filter/correlate below
        seq_orig = np.tile(seq_tmp, seq_num)
        seq_flip = seq_orig[::-1]  # flip
        # conjugate the local sequence
        seq_conj = np.conjugate(seq_flip)

        pool = mp.Pool(mp.cpu_count())
        m_filt = np.empty(
            [n_frame, n_cell, n_ue, n_ant, n_cmpx], dtype='complex64')
        indexing_start = time.time()
        for j in range(n_cell):
            result_objects = [pool.apply_async(epd.extract_pilots_data,
                                               args=(cmpx_pilots[i, j, :, :, :], seq_conj, seq_num, seq_len, i)) for i
                              in range(n_frame)]
            for i in range(n_frame):
                m_filt[i, j, :, :, :] = result_objects[i].get()[2]
        indexing_end = time.time()
        pool.close()

        return m_filt, seq_num, seq_len, cmpx_pilots, seq_orig

    def log2csi_hdf5(self, filename, offset=0):
        """Convert raw IQ log to CSI.

        Converts input Argos HDF5 trace to frequency domain CSI and writes it to the same filename with -csi appended.

        Args: filename of log

        Returns: None

        """
        print("inside log2csi")
        if legacy:
            h5log = h5py.File(filename,'r')
            symbol_len = h5log.attrs['samples_per_user']
            num_cl = h5log.attrs['num_mob_ant'] + 1
            pilot_samples = h5log['Pilot_Samples']
        else:
            symbol_len = int(self.h5file['Data'].attrs['SYMBOL_LEN'])
            num_cl = int(self.h5file['Data'].attrs['CL_NUM'])

        #compute CSI for each user and get a nice numpy array
        #Returns csi with Frame, User, LTS (there are 2), BS ant, Subcarrier
        #also, iq samples nic(Last 'user' is noise.)ely chunked out, same dims, but subcarrier is sample.
        csi,iq = samps2csi(pilot_samples, num_cl, symbol_len, offset=offset)

        # create hdf5 file to dump csi to
        h5f = h5py.File(filename[:-5]+'-csi.hdf5', 'w')
        h5f.create_dataset('csi', data=csi)
        #todo: copy gains
        #todo check if file exists os.path.isfile(fname) if f in glob.glob(f):
        for k in h5log.attrs:
                h5f.attrs[k] = h5log.attrs[k]
        h5f.close()
        h5log.close()
        print("finished log2csi")

    @staticmethod
    def samps2csi(samps, num_users, samps_per_user=224, fft_size=64, offset=0, bound=94, cp=0, sub=1, legacy=False, pilot_f=[]):
        """Convert an Argos HDF5 log file with raw IQ in to CSI.
        Asumes 802.11 style LTS used for trace collection.
    
        Args:
            samps: The h5py or numpy array containing the raw IQ samples,
                dims = [Frame, Cell, User, Antenna, Sample].
            num_users: Number of users used in trace collection. (Last 'user' is noise.)
            samps_per_user: Number of samples allocated to each user in each frame.
     
        Returns:
            csi: Complex numpy array with [Frame, Cell, User, Pilot Rep, Antenna, Subcarrier]
            iq: Complex numpy array of raw IQ samples [Frame, Cell, User, Pilot Rep, Antenna, samples]
            same modulation?
        Example:
            h5log = h5py.File(filename,'r')
            csi,iq = samps2csi(h5log['Pilot_Samples'], h5log.attrs['num_mob_ant']+1, h5log.attrs['samples_per_user'])
        """
        print("starting samps2csi")
        debug = False
        chunkstart = time.time()
        samps2csi_start = time.time()

        if legacy:

            offset = 47
            usersamps = np.reshape(samps, (samps.shape[0], samps.shape[1], num_users, samps_per_user, 2))
            if debug:
                print(
                    "chunkstart = {}, usersamps.shape = {}, samps.shape = {}, samps_per_user = {}, iq.shape = {}".format(
                        chunkstart, usersamps.shape, samps.shape, samps_per_user, iq.shape))
            print("samps2csi checkpoint1 took %f seconds" % (time.time() - samps2csi_start))
            iq = np.empty((samps.shape[0], samps.shape[1], num_users, 2, fft_size), dtype='complex64')
            print("checkpoint2 took %f seconds" % (time.time() - samps2csi_start))
            for i in range(2):  # 2 seperate estimates
                iq[:, :, :, i, :] = (usersamps[:, :, :, offset + i * fft_size:offset + (i + 1) * fft_size, 0] +
                                     usersamps[:, :, :, offset + i * fft_size:offset + (i + 1) * fft_size,
                                     1] * 1j) * 2 ** -15
            print("checkpoint3 took %f seconds" % (time.time() - samps2csi_start))
            iq = iq.swapaxes(1,
                             2)  # this is trickery to keep numpy vectorized (fast), then keep the axis in the order we want
            iq = iq.swapaxes(2, 3)
            if debug:
                print("iq.shape after axes swapping: {}".format(iq.shape))
            print("checkpoint4 took %f seconds" % (time.time() - samps2csi_start))
            fftstart = time.time()
            # csi = np.fft.fftshift(np.fft.fft(iq, fft_size, 4),4)*lts.lts_freq
            # a = pyfftw.empty_aligned(*iq.shape, dtype='complex64')
            # a[:] = iq
            # fft_object = pyfftw.builders.fft(a)
            # csi = fft_object()

            # csi = fft_a
            csi = np.empty(iq.shape, dtype='complex64')
            # pre_csi = np.fft.fftshift(np.fft.fft(iq, fft_size, 4),4)

            csi = np.fft.fftshift(np.fft.fft(iq, fft_size, 4), 4) * lts.lts_freq

            # pyfftw.interfaces.cache.enable()
            # csi = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fft(iq, fft_size, 4),4)*lts.lts_freq
            endtime = time.time()
            # print("checkpoint5 took %f seconds" % (samps2csi_start - time.time()))
            print("Chunk time: %f fft time: %f" % (fftstart - chunkstart, endtime - fftstart))
            csi = np.delete(csi, [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63], 4)  # remove zero subcarriers
            # print("samps2csi took %f seconds" % (samps2csi_start - time.time()))

        else:
            samps = samps[::sub]
            usersamps = np.reshape(
                samps, (samps.shape[0], samps.shape[1], num_users, samps.shape[3], samps_per_user, 2))
            pilot_rep = (samps_per_user-bound)//(fft_size+cp)
            #pilot_rep = min([(samps_per_user-bound)//(fft_size+cp), 2]) # consider min. 2 pilot reps
            iq = np.empty((samps.shape[0], samps.shape[1], num_users,
                           samps.shape[3], pilot_rep, fft_size), dtype='complex64')
            if debug:
                print("chunkstart = {}, usersamps.shape = {}, samps.shape = {}, samps_per_user = {}, iq.shape = {}".format(
                    chunkstart, usersamps.shape, samps.shape, samps_per_user, iq.shape))
            for i in range(pilot_rep):
                iq[:, :, :, :, i, :] = (usersamps[:, :, :, :, offset + cp + i*fft_size:offset+cp+(i+1)*fft_size, 0] +
                                        usersamps[:, :, :, :, offset + cp + i*fft_size:offset+cp+(i+1)*fft_size, 1]*1j)*2**-15

            iq = iq.swapaxes(3, 4)
            if debug:
                print("iq.shape after axes swapping: {}".format(iq.shape))

            fftstart = time.time()
            csi = np.empty(iq.shape, dtype='complex64')
            zero_sc = np.where(pilot_f == 0)[0]
            print('zero subcarrier:', zero_sc)
            nonzero_sc_size = len(pilot_f) - len(zero_sc)

            fft_length = int(np.power(2, np.ceil(np.log2(nonzero_sc_size))))
            if fft_length != fft_size:
                print("Expected fftsize %d, given %d"%(fft_length, fft_size))
                return None, iq
            #start_i = int((fft_length - nonzero_sc_size) // 2)
            #stop_i = int(start_i + nonzero_sc_size)
            nonzero_sc = np.setdiff1d(range(fft_size), zero_sc)
            print('nonzero_sc:', nonzero_sc)
            print('nonzero_sc.shape', nonzero_sc.shape)
            iq_fft = np.fft.fft(iq, fft_size, 5)
            seq_freq_inv = 1 / pilot_f[nonzero_sc]
            csi = iq_fft[:, : , :, :, :, nonzero_sc] * seq_freq_inv

            # calculate the approximate SNR range to run the experiment
            nonzero_sub = iq_fft[:, : , :, :, :, nonzero_sc]
            zero_sub = iq_fft[:, :, :, :, :, zero_sc]

            power_nonzero = np.sum(nonzero_sub*nonzero_sub.conjugate())/len(nonzero_sc)
            power_zero = np.sum(zero_sub*zero_sub.conjugate())/len(zero_sc)

            approximate_SNR = 10*np.log10(np.real(power_nonzero/power_zero))

            print('the approximate SNR:', approximate_SNR)

            endtime = time.time()
            if debug:
                print("csi.shape:{} lts_freq.shape: {}, pre_csi.shape = {}".format(
                    csi.shape, lts_freq.shape, pre_csi.shape))
            if debug:
                print("chunk time: %f fft time: %f" %
                      (fftstart - chunkstart, endtime - fftstart))
            if debug:
                print("csi.shape:{}".format(csi.shape))
            print("samps2csi took %f seconds" % (time.time() - samps2csi_start))

        # print("=======   start saving the csi & iq data   =======")
        # print("Running dirctory {}".format(sys.argv[0]))
        # path = '/home/keng/RENEWLab/PYTHON/IrisUtils/csi_data/'
        # if not os.path.exists(path):
        #     print("No such path!!! Now create one!!!")
        #     os.makedirs(path)
        #     print("Path create succeed!!!")
        # np.save('{}csi_data.npy'.format(path), csi)
        # np.save('{}iq_data.npy'.format(path), iq)
        # print("=======   finish saving the csi & iq data   =======")
        return csi, iq

    @staticmethod
    def samps2csi_large(samps, num_users, samps_per_user=224, offset=47, chunk_size=1000, legacy=False):
        """Wrapper function for samps2csi_main for to speed up large logs by leveraging data-locality. Chunk_size may need to be adjusted based on your computer."""

        samps2csi_start = time.time()
        if samps.shape[0] > chunk_size:
                    # rather than memmap let's just increase swap... should be just as fast.
                    #csi = np.memmap(os.path.join(_here,'temp1.mymemmap'), dtype='complex64', mode='w+', shape=(samps.shape[0], num_users, 2, samps.shape[1],52))
                    #iq = np.memmap(os.path.join(_here,'temp2.mymemmap'), dtype='complex64', mode='w+', shape=(samps.shape[0], num_users, 2, samps.shape[1],64))
            csi = np.empty(
                (samps.shape[0], num_users, 2, samps.shape[1], 52), dtype='complex64')
            iq = np.empty(
                (samps.shape[0], num_users, 2, samps.shape[1], 64), dtype='complex64')
            chunk_num = samps.shape[0]//chunk_size
            for i in range(chunk_num):
                csi[i*chunk_size:i*chunk_size+chunk_size], iq[i*chunk_size:i*chunk_size+chunk_size] = samps2csi(
                    samps[i*chunk_size:(i*chunk_size+chunk_size), :, :, :], num_users, samps_per_user=samps_per_user)
            csi[chunk_num*chunk_size:], iq[chunk_num*chunk_size:] = samps2csi(
                samps[chunk_num*chunk_size:, :, :, :], num_users, samps_per_user=samps_per_user)
        else:
            csi, iq = samps2csi(
                samps, num_users, samps_per_user=samps_per_user, offset=offset)

        print("samps2csi took %f seconds" % (samps2csi_start - time.time()))
        return csi, iq

    @staticmethod
    def frame_sanity(match_filt, k_lts, n_lts, st_frame = 0, frame_to_plot = 0, plt_ant=0, cp=16):
        """
        Creates a map of the frames per antenna. 3 categories: Good frames, bad frames, probably partial frames.
        Good frames are those where all k_lts peaks are present and spaced n_lts samples apart.
        Bad frames are those with random peaks.
        Potentially partial frames are those with some peaks at the right positions.
        This is a random event. Some frames may have accidentally some peaks at the right places.
        First the largest peak is detected, peaks at +1/-1 (probably due to multipath) and +CP/-CP samples are cleared out.
        Then, the positions of the largest k_lts peaks are determined.
        Finally, the function checks if these k_lts peaks are at the correct n_lts offstes.
        Disclaimer: This function is good only for a high SNR scenario!
        """

        debug = False
        dtct_eal_tx = True                  # Detect early transmission: further processing if this is desired
        n_frame = match_filt.shape[0]       # no. of captured frames
        n_cell = match_filt.shape[1]        # no. of cells
        n_ue = match_filt.shape[2]          # no. of UEs
        n_ant = match_filt.shape[3]         # no. of BS antennas
        n_corr = match_filt.shape[4]        # no. of corr. samples

        if debug:
            print("frame_sanity(): n_frame = {}, n_cell = {}, n_ue = {}, n_ant = {}, n_corr = {}, k_lts = {}".format(
            n_frame, n_cell, n_ue, n_ant, n_corr, k_lts) )


        # clean up the matched filter of extra peaks:
        mf_amax = np.argmax(match_filt, axis = -1)
        base_arr = np.arange(0,k_lts*n_lts, n_lts)
        for i in range(n_frame):
            for j in range(n_cell):
                for k in range(n_ue):
                    for l in range(n_ant):
                        mfa = mf_amax[i,j,k,l]

                        # WZC: indexs where side peaks might be
                        side_idx = np.delete(np.arange(n_corr), (mfa - base_arr))

                       # NB: addition: try to detect early packets: TEST it!
                        if dtct_eal_tx:
                            sim_thrsh  = 0.95 # similarity threshold bewtween two consequitive peaks
                            for ik in range(k_lts):
                                mf_prev = match_filt[i,j,k,l, (mfa - n_lts) if (mfa - n_lts) >= 0 else 0]
                                if 1 - np.abs(match_filt[i,j,k,l, mfa] -  mf_prev)/match_filt[i,j,k,l, mfa] >= sim_thrsh:
                                    mfa = (mfa - n_lts) if (mfa - n_lts) >= 0 else 0
                                else:
                                    break
                        # NB: addition: Clean everything right of the largest peak.
                        match_filt[i,j,k,l, mfa+1:] = 0         # we don't care about the peaks after the largest.
                        # misleading peaks seem to apear at +- argmax and argmax -1/+1/+2 CP and 29-30
                        for m in range(base_arr.shape[0]):
                            adj_idx1 = (mfa - 1) - base_arr[m]
                            adj_idx2 = (mfa + 1) - base_arr[m]
                            cp_idx1 = (mfa + cp) - base_arr[m]
                            cp_idx2 = (mfa + 1  + cp) - base_arr[m]
                            cp_idx3 = (mfa + -1  + cp) - base_arr[m]
                            idx_30 = (mfa + 30) - base_arr[m]
                            idx_29 = (mfa + 29) - base_arr[m]

                            idx_tot = [adj_idx1, adj_idx2, cp_idx1, cp_idx2, cp_idx3, idx_30, idx_29]

                            # WZC: removing possible side peaks
                            for idx in idx_tot:
                                if idx in side_idx:
                                    match_filt[i,j,k,l, idx] = 0

                            # if adj_idx1 >= 0 and adj_idx2 >=0 and adj_idx2 < n_corr:
                            #     match_filt[i,j,k,l, adj_idx1 ] = 0
                            #     match_filt[i,j,k,l, adj_idx2 ] = 0
                            # if (cp_idx1 >=0) and (cp_idx1 < n_corr) and (cp_idx2 < n_corr) and (cp_idx3 < n_corr):
                            #     match_filt[i,j,k,l, cp_idx1 ] = 0
                            #     match_filt[i,j,k,l, cp_idx2 ] = 0
                            #     match_filt[i,j,k,l, cp_idx3 ] = 0
                            # if (idx_30 >=0) and (idx_30 < n_corr) and (idx_29 >=0) and (idx_29 < n_corr):
                            #     match_filt[i,j,k,l,idx_30] = 0
                            #     match_filt[i,j,k,l,idx_29] = 0

        # get the k_lts largest peaks and their position
        k_max = np.sort(match_filt, axis = -1)[:,:,:,:, -k_lts:]
        k_amax =np.argsort(match_filt, axis = -1)[:,:,:,:, -k_lts:]
        # If the frame is good, the largerst peak is at the last place of k_amax
        lst_pk_idx = np.expand_dims(k_amax[:,:,:,:,-1], axis = 4)
        lst_pk_idx = np.tile(lst_pk_idx, (1,1,1,1,base_arr.shape[0]))
        # create an array with indices n_lts apart from each other relative to lst_pk_idx
        pk_idx = lst_pk_idx - np.tile(base_arr[::-1], (n_frame, n_cell, n_ue, n_ant,1))
        #subtract. In case of a good frame their should only be zeros in every postion
        idx_diff = k_amax - pk_idx
        frame_map = (idx_diff == 0).astype(np.int)
        # count the 0 and non-zero elements and reshape to n_frame-by-n_ant
        frame_map = np.sum(frame_map, axis =-1)
        #NB:
        zetas = frame_map*n_lts
        f_st = mf_amax - zetas

        if debug:
            print("f_st = {}".format(f_st[frame_to_plot - st_frame,0,0,:]))
            print("frame_sanity(): Shape of k_max.shape = {}, k_amax.shape = {}, lst_pk_idx.shape = {}".format(
                    k_max.shape, k_amax.shape, lst_pk_idx.shape))
            print("frame_sanity(): k_amax = {}".format(k_amax))
            print("frame_sanity(): frame_map.shape = {}\n".format(frame_map.shape))
            print(k_amax[frame_to_plot - st_frame,0,0,plt_ant,:])
            print(idx_diff[frame_to_plot - st_frame,0,0,plt_ant,:])

        peak_map = np.copy(frame_map)

        # This only works when pilot is repeated
        # TODO: come up with condition for single peak
        if (k_lts > 1):
            frame_map[frame_map == 1] = -1
            frame_map[frame_map >= (k_lts -1)] = 1
            frame_map[frame_map > 1] = 0
            if debug:
                print("frame_sanity(): frame_map = \n{}".format(frame_map))
                print(frame_to_plot - st_frame)

            #print results:
            n_rf = frame_map.size
            n_gf = frame_map[frame_map == 1].size
            n_bf = frame_map[frame_map == -1].size
            n_pr = frame_map[frame_map == 0].size
            print("===================== frame_sanity(): frame status: ============")
            print("Out of total {} received frames: \nGood frames: {}\nBad frames: {}\nProbably Partially received or corrupt: {}".format(
                    n_rf, n_gf, n_bf, n_pr,))
            print("===================== ============================= ============")

        return match_filt, frame_map, f_st, peak_map
        # WZC: added peak_map for peak_number analysis
