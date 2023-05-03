import glob
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from evaluation.Evaluation_metrics import recall, False_Alarm_Rate, precision, F1_score, detection_delay

batch_size = 100
latent_dim = 2 # 隐变量取2维只是为了方便后面画图
intermediate_dim = 2
epochs = 50

x = Input(shape=(10,))
h = Dense(intermediate_dim, activation='relu')(x)

# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# 解码层，也就是生成器部分
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(10, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

def preprocess(data, fixed_t):
    del_idx = []
    for i in range (data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

def sliding_window(elements, window_size):
    if len(elements) <= window_size:
        return elements
    new = np.empty((0, window_size))
    for i in range(len(elements) - window_size + 1):
        new = np.vstack((new, elements[i:i+window_size]))
    return new

if __name__ == '__main__':
    folder = '../data3/*.npz'
    fixed_threshold = 1.5
    forgetting_factor = 0.9
    stabilisation_period = 20
    p = 10
    g_noise = 0.005

    error_margin = 604800 # 7 days
    no_CPs = 0
    no_preds = 0
    no_TPS = 0
    delays = []

    for i in glob.glob(folder):
        data = np.load(i, allow_pickle=True)
        name = i[-19:-12]
        train_ts, train_dl, test_ts_1gal, test_dl_1gal, label = data['train_ts'], data['train_dl'], data['test_ts_2gal'], data['test_dl_2gal'], data['label'].item()
        dl = np.concatenate((train_dl, test_dl_1gal))
        test_dl_1gal = test_dl_1gal[~np.isnan(test_dl_1gal).any(axis=1)]
        test_ts_1gal = test_ts_1gal[~np.isnan(test_ts_1gal).any(axis=1)]

        test_dl_1gal = preprocess(test_dl_1gal, fixed_threshold)
        test_ts_1gal = preprocess(test_ts_1gal, fixed_threshold)

        ts = test_dl_1gal[:, 0]
        cps = label['test_2gal']

        train_var_dl = train_dl[:, 1]
        train_ht_dl = train_dl[:, 2]
        test_var_dl = test_dl_1gal[:, 1]
        test_ht_dl = test_dl_1gal[:, 2]
        multi_test = np.stack((test_var_dl, test_ht_dl), axis=1)
        # test_var_dl = np.reshape(test_var_dl, (test_var_dl.shape[0], 1))

        # vae = VAE(encoder, decoder)
        # vae.compile(loss=None,optimizer=keras.optimizers.Adam())
        ctr = 0
        List_st = []  # list of timestamps
        buffer_ts = 500  # Number of timestamps for initilisation
        cpcands = []
        w = 10  # window size
        scores = np.zeros(test_var_dl.shape[0])

        # 建立模型
        vae = Model(x, x_decoded_mean)

        # xent_loss是重构loss，kl_loss是KL loss
        xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)

        # add_loss是新增的方法，用于更灵活地添加各种loss
        vae.add_loss(vae_loss)
        vae.compile(optimizer='rmsprop')
        vae.summary()

        while ctr < test_var_dl.shape[0]:
            List_st.append(ctr)
            if len(List_st) == buffer_ts: # initialise CPD model
                data = test_var_dl[List_st]
                noise = np.random.normal(0, g_noise, size=(data.shape[0]))
                noisy_Data = data + noise

                noisy_Data = sliding_window(noisy_Data, w)
                data = sliding_window(data, w)

                vae.fit(noisy_Data, epochs=20, batch_size=32, shuffle=True)
                MAE = np.mean(np.abs(vae.predict(noisy_Data) - data))
                threshold = 0.9 * MAE
            elif len(List_st) > buffer_ts:
                window = np.reshape(test_var_dl[ctr-9:ctr+1],(1,10))
                pred = vae.predict(window)
                score = np.mean(np.abs(vae.predict(window) - window))
                scores[ctr] = score
                if score > threshold:
                    cpcands.append(ctr)
                    List_st = []
            ctr += 1

        no_CPs += len(cps)
        no_preds += len(cpcands)
        for j in cpcands:
            timestamp = ts[j]
            for l in cps:
                if timestamp >= l and timestamp <= l + error_margin:
                    no_TPS += 1
                    delays.append(timestamp - l)

        fig = plt.figure()
        fig, ax = plt.subplots(2, figsize=[18, 16], sharex=True)
        ax[0].plot(ts, multi_test[:, 0])
        for cp in cps:
            ax[0].axvline(x=cp, color='g', alpha=0.6)

        ax[1].plot(ts, scores)
        for cp in cpcands:
            ax[1].axvline(x=ts[cp], color='g', alpha=0.6)

        plt.savefig(name + '.png')
    rec = recall(no_TPS, no_CPs)
    FAR = False_Alarm_Rate(no_preds, no_TPS)
    prec = precision(no_TPS, no_preds)
    f1score = F1_score(rec, prec)
    dd = detection_delay(delays)
    print('recall: ', rec)
    print('false alarm rate: ', FAR)
    print('precision: ', prec)
    print('F1 Score: ', f1score)
    print('detection delay: ', dd)