import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

result_dir = '../results/'

online_tr4_re32_QAM16 = result_dir + 'OnlineTraining_QAM_16_NT4NR32_MMNet/'

# learning based online transmit*4 receive*32 QAM16
acc = np.array([])

for num in range(100):
    data = pd.read_pickle(online_tr4_re32_QAM16 + f'results{num}.pkl')
    for key in data.keys():
        if not key == 'cond':
            coordinate = data[key]
            acc = np.append(acc, [coordinate['accuracy']], axis=0)

acc = acc.reshape(100, -1)
mean_acc = acc.mean(axis=0)
MMNet_SER_QAM16 = 1 - mean_acc

# change SER into log scale

plt.title('QAM_4 with tr32_re64')
plt.xlabel('SNR(dB)')
plt.ylabel('SER')
plt.semilogy()
MMNet_16, = plt.plot(range(10,16), MMNet_SER_QAM16, color='k')
plt.legend([MMNet_16], ["MMNet"], loc = 'center right')
plt.grid(b=True, which='major', axis='y')
plt.show()
