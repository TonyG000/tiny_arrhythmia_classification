# data_generator_file
import numpy as np
import tensorflow as tf
from data_processing.cpsc_signal_loader import RawSignalLoader, SignalProcessor
from config import LEAD_MODE, DESIRED_LEAD, BATCH_SIZE, SAMPLING_RATE, TARGET_LENGTH, DATASET_PATH

class DataGenerator(tf.keras.utils.Sequence):
    """Keras-compatible data generator for ECG signals."""

    def __init__(self, df, label_encoder, shuffle=True):
        self.df = df
        self.label_encoder = label_encoder
        self.batch_size = BATCH_SIZE
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.sampling_rate = SAMPLING_RATE
        self.target_length = TARGET_LENGTH
        self.desired_lead = DESIRED_LEAD
        self.lead_mode = LEAD_MODE

        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        """Generates a batch of data."""
        batch_indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(batch_indices)

    def __data_generation(self, batch_indices):
        """Loads and preprocesses batch data."""
        batch_signals = []
        batch_labels = []

        # Initialize SignalProcessor
        signal_processor = SignalProcessor(sampling_rate=self.sampling_rate, target_length=self.target_length, lead_mode=self.lead_mode, desired_lead=self.desired_lead)

        for idx in batch_indices:
            # file_path = DATASET_PATH + self.df.iloc[idx]['file_path']
            # file_path = DATASET_PATH + self.df.iloc[idx]['file_path']
            file_path = self.df.iloc[idx]['file_path']
            
            # Load raw signal using RawSignalLoader
            raw_loader = RawSignalLoader(file_path=file_path, lead_mode=self.lead_mode, desired_lead=self.desired_lead)
            raw_signals, _ = raw_loader.load_signal()
            
            # Process the raw signal using SignalProcessor
            processed_signals = signal_processor.process_signal(raw_signals)
            batch_signals.append(processed_signals)
            
            # Get label
            label = self.df.iloc[idx]['classes']
            batch_labels.append(self.label_encoder.transform([label])[0])

        return (
            np.array(batch_signals, dtype=np.float32),
            np.array(batch_labels, dtype=np.int32)
        )

    def on_epoch_end(self):
        """Shuffles indexes after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)
