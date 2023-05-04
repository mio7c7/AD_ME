import glob
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import math
sys.path.append('./')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA

parser = argparse.ArgumentParser(description='Mstatistics evaluation on bottom 0.2 data')
parser.add_argument('--data', type=str, default='../data3/*.npz', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--g_noise', type=float, default=0.005, help='gaussian noise')
parser.add_argument('--buffer_ts', type=int, default=500, help='Number of timestamps for initilisation')
parser.add_argument('--bs', type=int, default=24, help='buffer size for ssa')
parser.add_argument('--latent_dim', type=int, default=2, help='threshold')
parser.add_argument('--batch_size', type=int, default=32, help='threshold')
parser.add_argument('--epoch', type=int, default=20, help='epoch')
parser.add_argument('--threshold', type=float, default=0.9, help='threshold')
parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
parser.add_argument('--outfile', type=str, default='AE', help='name of file to save results')
args = parser.parse_args()

latent_dim = args.latent_dim # 隐变量取2维只是为了方便后面画图
intermediate_dim = 2

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
    folder = args.data
    fixed_threshold = 1.5

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

        ssa = SSA(window=args.ssa_window, max_length=args.buffer_ts)
        ctr = args.buffer_ts
        X = test_var_dl[:ctr]
        X_pred = ssa.reset(X)
        X_pred = ssa.transform(X_pred, state=ssa.get_state())
        reconstructeds = X_pred.sum(axis=0)
        residuals = X - reconstructeds
        resmean = residuals.mean()
        M2 = ((residuals - resmean) ** 2).sum() * (len(residuals) - 1) * residuals.var()

        # vae = VAE(encoder, decoder)
        # vae.compile(loss=None,optimizer=keras.optimizers.Adam())
        List_st = [i for i in range(ctr)]  # list of timestamps
        preds = []
        w = 10  # window size
        step = args.bs
        scores = np.zeros(test_var_dl.shape[0])
        outliers = []
        es = EarlyStopping(patience=5, verbose=1, min_delta=0.001, monitor='val_loss', mode='auto',
                           restore_best_weights=True)
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
        # initialise CPD model
        noise = np.random.normal(0, args.g_noise, size=(X.shape[0]))
        noisy_Data = X + noise
        noisy_Data = sliding_window(noisy_Data, w)
        X = sliding_window(X, w)

        vae.fit(noisy_Data, epochs=args.epoch, batch_size= args.batch_size, shuffle=True, validation_split=0.2, callbacks=[es])
        MAE = np.mean(np.abs(vae.predict(noisy_Data) - X))
        threshold = args.threshold * MAE

        while ctr < test_var_dl.shape[0]:
            new = test_var_dl[ctr:ctr + step]
            updates = ssa.update(new)
            updates = ssa.transform(updates, state=ssa.get_state())[:, -step:]
            reconstructed = updates.sum(axis=0)
            residual = new - reconstructed
            residuals = np.concatenate([residuals, residual])
            reconstructeds = np.concatenate((reconstructeds, reconstructed))

            for k in range(len(new)):
                delta = residual[k] - resmean
                resmean += delta / (ctr + k)
                M2 += delta * (residual[k] - resmean)
                stdev = math.sqrt(M2 / (ctr + k - 1))
                threshold_upper = resmean + 2 * stdev
                threshold_lower = resmean - 2 * stdev

                if residual[k] > threshold_upper or residual[k] < threshold_lower:
                    outliers.append(ctr + k)
                    continue

                List_st.append(ctr+k)
                if len(List_st) == args.buffer_ts:
                    X = reconstructeds[List_st]
                    noise = np.random.normal(0, args.g_noise, size=(X.shape[0]))
                    noisy_Data = X + noise
                    noisy_Data = sliding_window(noisy_Data, w)
                    X = sliding_window(X, w)
                    vae.fit(noisy_Data, epochs=args.epoch, batch_size=args.batch_size, shuffle=True,
                            validation_split=0.2, callbacks=[es])
                    MAE = np.mean(np.abs(vae.predict(noisy_Data) - X))
                    threshold = args.threshold * MAE
                elif len(List_st) > args.buffer_ts:
                    window = np.reshape(reconstructeds[ctr-9:ctr+1],(1,10))
                    pred = vae.predict(window)
                    score = np.mean(np.abs(pred - window))
                    scores[ctr] = score
                    if score > threshold:
                        preds.append(ctr)
                        List_st = []

            if len(test_var_dl) - ctr <= args.bs:
                break
            elif len(test_var_dl) - ctr <= 2 * args.bs:
                ctr += args.bs
                step = len(test_var_dl) - ctr
                print(step)
            else:
                ctr += args.bs

        no_CPs += len(cps)
        no_preds += len(preds)
        for j in preds:
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
        for cp in preds:
            ax[1].axvline(x=ts[cp], color='g', alpha=0.6)

        plt.savefig(name + '.png')

    rec = Evaluation_metrics.recall(no_TPS, no_CPs)
    FAR = Evaluation_metrics.False_Alarm_Rate(no_preds, no_TPS)
    prec = Evaluation_metrics.precision(no_TPS, no_preds)
    f1score = Evaluation_metrics.F1_score(rec, prec)
    dd = Evaluation_metrics.detection_delay(delays)
    print('recall: ', rec)
    print('false alarm rate: ', FAR)
    print('precision: ', prec)
    print('F1 Score: ', f1score)
    print('detection delay: ', dd)

    npz_filename = args.outfile
    np.savez(npz_filename,
             rec=rec, FAR=FAR, prec=prec, f1score=f1score, dd=dd)