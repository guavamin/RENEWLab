import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops import gen_math_ops
import os
from tf_session import *
import pickle
from parser import parse
from exp import get_data
import time

start = time.time()

params, args = parse()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
num_channel_samples = 100

def complex_to_real(inp):
    Hr = np.real(inp)
    Hi = np.imag(inp)
    h1 = np.concatenate([Hr, -Hi], axis=2)
    # print('h1.shape:', h1.shape)
    h2 = np.concatenate([Hi,  Hr], axis=2)
    # print('h2.shape:', h2.shape)
    out = np.concatenate([h1, h2], axis=1)
    # print('out.shape:', out.shape)
    return out

def complex_to_real_for_uplink(inp):
    Hr = np.real(inp)
    # print('Hr.shape:', Hr.shape)
    Hi = np.imag(inp)
    # print('Hi.shape:', Hi.shape)
    out = np.concatenate([Hr, Hi], axis=1)
    # print('out.shape:', out.shape)
    return out

if args.data:
    H_dataset = np.load(args.channels_dir)
    print('H_dataset.shape:', H_dataset.shape)
    assert (H_dataset.shape[-3] == args.x_size), print(args.x_size)
    assert (H_dataset.shape[-2] == args.y_size), print(args.y_size)

    # H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-3]))
    # print(H_dataset.shape)
    H_dataset = complex_to_real(H_dataset)
    Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.)
    params['Hdataset_powerdB'] = Hdataset_powerdB
    print('Channels dataset power (dB): %f'%Hdataset_powerdB)

    # train_data_ref = H_dataset
    # test_data_ref = H_dataset
    # rndIndx = np.random.randint(0, train_data_ref.shape[0], num_channel_samples)
    # print('rndIndx:', rndIndx)

    #
    H_dataset = H_dataset.transpose((0,2,1,3))
    train_data_ref = H_dataset[:, :, :, 0]
    test_data_ref = H_dataset

    # print('train_data_ref :', train_data_ref)
    # print('test_data_ref :', test_data_ref)

    # print('Sampled channel indices: ', rndIndx)
    # print('Sampled channel indices.shape: ', rndIndx.shape)


else:
    test_data = []
    train_data = []

# print("Debug Testing")

# Build the computational graph
mmnet = MMNet_graph(params)
nodes = mmnet.build()

# Get access to the nodes on the graph
sess = nodes['sess']
x = nodes['x']
H = nodes['H']
x_id = nodes['x_id']
constellation = nodes['constellation']
train = nodes['train']
snr_db_min = nodes['snr_db_min']
snr_db_max = nodes['snr_db_max']
lr = nodes['lr']
batch_size = nodes['batch_size']
accuracy = nodes['accuracy']
mmse_accuracy = nodes['mmse_accuracy']
loss = nodes['loss']
logs = nodes['logs']
measured_snr = nodes['measured_snr']
init = nodes['init']

uplink_data = np.load('{}uplink_data.npy'.format(data_path))
uplink_data = uplink_data[:,:,0,:]
np.squeeze(uplink_data)
uplink_data = complex_to_real_for_uplink(uplink_data)
# Training loop
sess.run(init)
results = {}
for it in range(args.train_iterations+1):
    feed_dict = {
                batch_size: args.batch_size,
                lr: args.learn_rate,
                snr_db_max: params['SNR_dB_max'],
                snr_db_min: params['SNR_dB_min'],
                x: uplink_data[:,:,0]
            }
    if args.data:
        # print('np.shape(train_data)[0]', np.shape(train_data)[0])
        # sample_ids = np.random.randint(0, np.shape(train_data)[0], params['batch_size'])
        # print('sample_ids', sample_ids)
        feed_dict[H] = train_data_ref
        # print('feed_dict[H].shape:', feed_dict[H].shape)

    sess.run(train, feed_dict)

    # Test
    if it == args.train_iterations:
        for subcarrier_num in range(H_dataset.shape[3]):
            for snr_ in range(int(params['SNR_dB_min']), int(params['SNR_dB_max'])+1):
                feed_dict = {
                        batch_size: args.test_batch_size,
                        snr_db_max: snr_,
                        snr_db_min: snr_,
                        x: uplink_data[:,:,subcarrier_num]
                    }
                if args.data:
                    # sample_ids = np.random.randint(0, np.shape(test_data)[0], args.test_batch_size)
                    # print('sample_ids:', sample_ids)
                    feed_dict[H] = test_data_ref[:, :, :, subcarrier_num]

                test_accuracy_, test_loss_, measured_snr_, log_ = sess.run([accuracy, loss, measured_snr, logs], feed_dict)
                print('Test SER of %f on different subcarrier realization %d after %d iterations at SNR %f dB'%(1. - test_accuracy_, subcarrier_num, it, measured_snr_))
                results[str(snr_)] = {}
                for k in log_:
                    results[str(snr_)][k] = log_[k]['stat']
                results[str(snr_)]['accuracy'] = test_accuracy_
            results['cond'] = np.linalg.cond(test_data_ref[:, :, :, subcarrier_num])
            path = args.output_dir+'/OnlineTraining_%s_NT%sNR%s_%s/'%(args.modulation, args.x_size, args.y_size, args.linear)
            if not os.path.exists(path):
                os.makedirs(path)
            savePath = path+'results%d.pkl'%subcarrier_num
            with open(savePath, 'wb') as f:
                pickle.dump(results, f)
            print('Results saved at %s'%savePath)

end = time.time()

print(">>>>>>> all the time needed = {} seconds".format(end - start))
