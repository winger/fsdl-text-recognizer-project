from typing import Callable, Dict
import pathlib

from boltons.cacheutils import cachedproperty
import numpy as np
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam, RMSprop

from text_recognizer.datasets.base import Dataset
from text_recognizer.datasets.sequence import DatasetSequence


DIRNAME = pathlib.Path(__file__).parents[1].resolve() / 'weights'


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""
    def __init__(self, dataset_cls: type, network_fn: Callable, dataset_args: Dict=None, network_args: Dict=None):
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)
        self.network.summary()

        self.batch_augment_fn = None
        self.batch_format_fn = None

    @property
    def weights_filename(self):
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.h5')

    def fit(self, dataset, batch_size=32, epochs=10, callbacks=[]):
        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())

        train_sequence = DatasetSequence(dataset.x_train, dataset.y_train, batch_size, augment_fn=self.batch_augment_fn, format_fn=self.batch_format_fn)
        test_sequence = DatasetSequence(dataset.x_test, dataset.y_test, batch_size, augment_fn=self.batch_augment_fn, format_fn=self.batch_format_fn)

        self.network.fit_generator(
            train_sequence,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=test_sequence,
            use_multiprocessing=True,
            workers=1,
            shuffle=True
        )

    def evaluate(self, x, y):
        sequence = DatasetSequence(x, y, batch_size=16)  # Use a small batch size to use less memory
        preds = self.network.predict_generator(sequence)
        return np.mean(np.argmax(preds, -1) == np.argmax(y, -1))

    def loss(self):
        return 'categorical_crossentropy'

    def optimizer(self):
        return Adam(3e-4)

    def metrics(self):
        return ['accuracy']

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)
