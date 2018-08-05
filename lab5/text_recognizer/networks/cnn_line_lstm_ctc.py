from boltons.cacheutils import cachedproperty
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Add, Activation, BatchNormalization, Concatenate, Conv1D, Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU, MaxPooling2D, Permute, RepeatVector, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, CuDNNGRU, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.ctc import ctc_decode


def cnn_line_lstm_ctc(input_shape, output_shape, **kwargs):
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    image_input = Input(shape=input_shape, name='image')
    y_true = Input(shape=(output_length,), name='y_true')
    input_length = Input(shape=(1,), name='input_length')
    label_length = Input(shape=(1,), name='label_length')

    gpu_present = len(device_lib.list_local_devices()) > 1
    lstm_fn = CuDNNLSTM if gpu_present else LSTM

    ##### Your code below (Lab 3)
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    # (image_height, image_width, 1)

    convnet_outputs = image_reshaped
    # convnet_outputs = Dropout(0.5)(convnet_outputs)
    convnet_outputs = Conv2D(16, 3, padding='SAME')(convnet_outputs)
    convnet_outputs = BatchNormalization()(convnet_outputs)
    convnet_outputs = LeakyReLU()(convnet_outputs)
    convnet_outputs = MaxPooling2D(2, 2)(convnet_outputs)

    # convnet_outputs = Dropout(0.5)(convnet_outputs)
    convnet_outputs = Conv2D(32, 3, padding='SAME')(convnet_outputs)
    convnet_outputs = BatchNormalization()(convnet_outputs)
    convnet_outputs = LeakyReLU()(convnet_outputs)
    convnet_outputs = MaxPooling2D(2, 2)(convnet_outputs)

    convnet_outputs = Dropout(0.2)(convnet_outputs)
    convnet_outputs = Conv2D(48, 3, padding='SAME')(convnet_outputs)
    convnet_outputs = BatchNormalization()(convnet_outputs)
    convnet_outputs = LeakyReLU()(convnet_outputs)
    convnet_outputs = MaxPooling2D(2, 2)(convnet_outputs)

    convnet_outputs = Dropout(0.2)(convnet_outputs)
    convnet_outputs = Conv2D(64, 3, padding='SAME')(convnet_outputs)
    convnet_outputs = BatchNormalization()(convnet_outputs)
    convnet_outputs = LeakyReLU()(convnet_outputs)

    convnet_outputs = Dropout(0.2)(convnet_outputs)
    convnet_outputs = Conv2D(80, 3, padding='SAME')(convnet_outputs)
    convnet_outputs = BatchNormalization()(convnet_outputs)
    convnet_outputs = LeakyReLU()(convnet_outputs)

    num_windows = 119

    convnet_outputs = Permute([2, 1, 3])(convnet_outputs)
    convnet_outputs = Reshape([num_windows, 240])(convnet_outputs)

    # (num_windows, 128)

    lstm_output = convnet_outputs
    for i in range(2):
        lstm_output = Dropout(0.5)(lstm_output)
        lstm_output = Bidirectional(lstm_fn(256, return_sequences=True))(lstm_output)
    lstm_output = Dropout(0.5)(lstm_output)

    softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output)
    # (num_windows, num_classes)
    ##### Your code above (Lab 3)

    input_length_processed = Lambda(
        lambda x, num_windows=None: x * num_windows,
        arguments={'num_windows': num_windows}
    )(input_length)

    ctc_loss_output = Lambda(
        lambda x: K.ctc_batch_cost(x[0], x[1], x[2], x[3]),
        name='ctc_loss'
    )([y_true, softmax_output, input_length_processed, label_length])

    ctc_decoded_output = Lambda(
        lambda x: ctc_decode(x[0], x[1], output_length),
        name='ctc_decoded'
    )([softmax_output, input_length_processed])

    model = KerasModel(
        inputs=[image_input, y_true, input_length, label_length],
        outputs=[ctc_loss_output, ctc_decoded_output]
    )
    return model
