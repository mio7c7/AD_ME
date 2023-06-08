import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.compat.v1.keras.layers import *
from tensorflow.compat.v1.keras.models import *
from tensorflow.compat.v1.keras.optimizers import *
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.metrics import *
from tensorflow.compat.v1.keras import regularizers
# Class Sampling
loss_metric = Mean(name='loss')
recon_metric = Mean(name='recon_loss')
class Encoder(Model):
    def __init__(self, timestep, input_dim, hid_dim, activation, dropout, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.encoder_inputs = Input(shape=(timestep, input_dim), name='Input')
        self.encoder = Dense(hid_dim, activation=activation)
        self.flat = Flatten()
        self.dropout = Dropout(dropout)

    def call(self, inputs):
        self.encoder_inputs = inputs
        flat = self.flat(self.encoder_inputs)
        hidden = self.encoder(flat)
        z = self.dropout(hidden)
        return z

class Decoder(Layer):
    def __init__(self, timestep, input_dim, hid_dim, activation, dropout, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.decoder = Dense(hid_dim, activation=activation)
        self.dec = Dense(timestep, activation='sigmoid')
        self.reshape = Reshape((timestep, input_dim))
        self.dropout = Dropout(dropout)

    def call(self, inputs):
        hidden =self.decoder(inputs)
        hidden = self.dropout(hidden)
        pred = self.reshape(self.dec(hidden))
        return pred

# Define VAE as a model

class AE(Model):
    def __init__(self, timestep, input_dim, lstm_dim, activation, dropout, name='vae', **kwargs):
        super(AE, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(timestep, input_dim, lstm_dim, activation, dropout,**kwargs)
        self.decoder = Decoder(timestep, input_dim, lstm_dim, activation, dropout,**kwargs)
        self.timestep = timestep

    def call(self, inputs):
        z = self.encoder(inputs)
        pred = self.decoder(z)
        return z, pred

    def reconstruct_loss(self, inputs, pred):
        return K.mean(K.sum(K.binary_crossentropy(inputs, pred), axis=-1))
        # return K.mean(K.square(inputs - pred))

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            z, pred = self(inputs, training=True)
            reconstruction = self.reconstruct_loss(inputs, pred)
            loss = K.mean(reconstruction)
            loss += sum(self.losses)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # print(loss)
        loss_metric.update_state(loss)
        recon_metric.update_state(reconstruction)
        return {'loss': loss_metric.result(), 'rec_loss': recon_metric.result()}

    def test_step(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        z, pred = self(inputs, training=True)
        reconstruction = self.reconstruct_loss(inputs, pred)
        total_loss = K.mean(reconstruction)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction,
        }
