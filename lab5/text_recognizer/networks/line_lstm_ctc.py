from boltons.cacheutils import cachedproperty
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Add, Activation, BatchNormalization, Concatenate, Conv1D, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, RepeatVector, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, CuDNNGRU, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window, slide_window_flatten
from text_recognizer.networks.ctc import ctc_decode


def line_lstm_ctc(input_shape, output_shape, **kwargs):
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    image_input = Input(shape=input_shape, name='image')
    y_true = Input(shape=(output_length,), name='y_true')
    input_length = Input(shape=(1,), name='input_length')
    label_length = Input(shape=(1,), name='label_length')

    gpu_present = len(device_lib.list_local_devices()) > 1
    lstm_fn = CuDNNLSTM if gpu_present else LSTM

    # Your code should use slide_window and extract image patches from image_input.
    # Pass a convolutional model over each image patch to generate a feature vector per window.
    # Pass these features through one or more LSTM layers.
    # Convert the lstm outputs to softmax outputs.
    # Note that lstms expect a input of shape (num_batch_size, num_timesteps, feature_length).

    ##### Your code below (Lab 3)
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    # (image_height, image_width, 1)

    convnet_outputs = image_reshaped
    convnet_outputs = BatchNormalization()(convnet_outputs)
    # convnet_outputs = Dropout(0.2)(convnet_outputs)
    convnet_outputs = Conv2D(32, kernel_size=(3, 3), activation='relu')(convnet_outputs)
    # convnet_outputs = Dropout(0.2)(convnet_outputs)
    convnet_outputs = BatchNormalization()(convnet_outputs)
    convnet_outputs = Conv2D(64, (3, 3), activation='relu')(convnet_outputs)
    # convnet_outputs = Dropout(0.2)(convnet_outputs)
    convnet_outputs = MaxPooling2D(pool_size=(2, 2))(convnet_outputs)
    convnet_outputs = Dropout(0.5)(convnet_outputs)
    # convnet_outputs = MaxPooling2D(pool_size=(12, 1))(convnet_outputs)
    convnet_outputs = Lambda(
        slide_window_flatten,
        arguments={'window_width': 12, 'window_stride': 1}
    )(convnet_outputs)
    convnet_outputs = Dense(128, activation='relu')(convnet_outputs)
    print(convnet_outputs)
    num_windows = 463

    # (num_windows, 128)

    lstm_output = Dropout(0.5)(convnet_outputs)
    for i in range(kwargs.get('lstm_layers', 1)):
        # lstm_output = Bidirectional(lstm_fn(256, return_sequences=True))(lstm_output)
        # lstm_output = Dropout(0.5)(lstm_output)
        lstm_output = BatchNormalization()(lstm_output)
        lstm_output = Conv1D(256, 3, activation='relu', padding='SAME')(lstm_output)
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
