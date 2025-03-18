import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Activation, 
    Multiply, Lambda, Conv1D, Layer
)
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

class InstantLayerNormalization(Layer):
    '''
    Class implementing instant layer normalization. It can also be called
    channel-wise layer normalization and was proposed by
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
    '''

    def __init__(self, **kwargs):
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True,
            name='gamma'
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='beta'
        )

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
        std = tf.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs = outputs * self.gamma + self.beta
        return outputs


class DTLN_model:
    '''
    Class to create and train the DTLN model.
    '''

    def __init__(self):
        # Default parameters
        self.fs = 16000
        self.batchsize = 32
        self.len_samples = 15
        self.activation = 'sigmoid'
        self.numUnits = 128
        self.numLayer = 2
        self.blockLen = 512
        self.block_shift = 128
        self.dropout = 0.25
        self.lr = 1e-3
        self.max_epochs = 200
        self.encoder_size = 256
        self.eps = 1e-7

        # Set seeds for reproducibility
        os.environ['PYTHONHASHSEED'] = '42'
        np.random.seed(42)
        tf.random.set_seed(42)

        # Enable GPU memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

    @staticmethod
    def snr_cost(s_estimate, s_true):
        '''
        Static Method defining the cost function.
        The negative signal to noise ratio is calculated here.
        '''
        snr = tf.reduce_mean(tf.square(s_true), axis=-1, keepdims=True) / \
              (tf.reduce_mean(tf.square(s_true - s_estimate), axis=-1, keepdims=True) + 1e-7)
        num = tf.math.log(snr)
        denom = tf.math.log(tf.constant(10, dtype=num.dtype))
        loss = -10 * (num / denom)
        return loss

    def lossWrapper(self):
        '''
        A wrapper function which returns the loss function.
        '''
        def lossFunction(y_true, y_pred):
            loss = tf.squeeze(self.snr_cost(y_pred, y_true))
            loss = tf.reduce_mean(loss)
            return loss
        return lossFunction

    def stftLayer(self, x):
        '''
        Method for an STFT helper layer.
        '''
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        stft_dat = tf.signal.rfft(frames)
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        return [mag, phase]

    def ifftLayer(self, x):
        '''
        Method for an inverse FFT layer.
        '''
        s1_stft = tf.cast(x[0], tf.complex64) * tf.exp((1j * tf.cast(x[1], tf.complex64)))
        return tf.signal.irfft(s1_stft)

    def overlapAddLayer(self, x):
        '''
        Method for an overlap and add helper layer.
        '''
        return tf.signal.overlap_and_add(x, self.block_shift)

    def seperation_kernel(self, num_layer, mask_size, x, stateful=False):
        '''
        Method to create a separation kernel.
        '''
        for idx in range(num_layer):
            x = LSTM(self.numUnits, return_sequences=True, stateful=stateful)(x)
            if idx < (num_layer - 1):
                x = Dropout(self.dropout)(x)
        mask = Dense(mask_size)(x)
        mask = Activation(self.activation)(mask)
        return mask

    def build_DTLN_model(self, norm_stft=False):
        '''
        Method to build and compile the DTLN model.
        '''
        # Input layer for time signal
        time_dat = Input(batch_shape=(None, None))

        # STFT and normalization
        mag, angle = Lambda(self.stftLayer)(time_dat)
        if norm_stft:
            mag_norm = InstantLayerNormalization()(tf.math.log(mag + 1e-7))
        else:
            mag_norm = mag

        # First separation core
        mask_1 = self.seperation_kernel(self.numLayer, (self.blockLen // 2 + 1), mag_norm)
        estimated_mag = Multiply()([mag, mask_1])
        estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag, angle])

        # Second separation core
        encoded_frames = Conv1D(self.encoder_size, 1, strides=1, use_bias=False)(estimated_frames_1)
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        mask_2 = self.seperation_kernel(self.numLayer, self.encoder_size, encoded_frames_norm)
        estimated = Multiply()([encoded_frames, mask_2])
        decoded_frames = Conv1D(self.blockLen, 1, padding='causal', use_bias=False)(estimated)
        estimated_sig = Lambda(self.overlapAddLayer)(decoded_frames)

        # Create the model
        self.model = Model(inputs=time_dat, outputs=estimated_sig)

    def compile_model(self):
        '''
        Method to compile the model for training.
        '''
        optimizerAdam = Adam(learning_rate=self.lr, clipnorm=3.0)
        self.model.compile(loss=self.lossWrapper(), optimizer=optimizerAdam)