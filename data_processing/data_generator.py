# data_generator_file
import numpy as np
import tensorflow as tf
from data_processing.cpsc_signal_loader import RawSignalLoader, SignalProcessor
from config import (
    LEAD_MODE, DESIRED_LEAD, BATCH_SIZE, SAMPLING_RATE, TARGET_LENGTH,
    DATASET_PATH, LEAD_MASKING,
    RARE_CLASS_AUG, RARE_AUG_NOISE_STD, RARE_AUG_MAX_SHIFT_FRAC,
)

# Lead counts that mirror wearable deployment scenarios (1-lead patch, 3-lead
# holter, 12-lead clinical). Masking to these during training teaches the model
# to produce robust representations regardless of how many leads are available.
_MASKING_LEAD_OPTIONS = [1, 3, 12, 12]  # 12-lead gets 50%, 1 and 3 get 25% each

class DataGenerator(tf.keras.utils.Sequence):
    """Keras-compatible data generator for ECG signals."""

    def __init__(self, df, label_encoder, shuffle=True, augment=False,
                 rare_class_indices=None):
        self.df = df
        self.label_encoder = label_encoder
        self.batch_size = BATCH_SIZE
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.df))
        self.sampling_rate = SAMPLING_RATE
        self.target_length = TARGET_LENGTH
        self.desired_lead = DESIRED_LEAD
        self.lead_mode = LEAD_MODE
        # Encoded label indices that should receive ECG-specific augmentation
        # (Gaussian noise + circular time shift) on top of lead masking.
        self.rare_class_indices = set(rare_class_indices or [])

        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        """Generates a batch of data."""
        batch_indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(batch_indices)

    def _random_lead_mask(self, signal):
        """Randomly keep 1, 3, or 12 leads; zero out the rest.

        signal shape: (time, leads)
        """
        if not LEAD_MASKING:
            return signal

        num_leads = signal.shape[-1]
        k = np.random.choice([opt for opt in _MASKING_LEAD_OPTIONS if opt <= num_leads])

        if k == num_leads:
            return signal

        kept_leads = np.random.choice(num_leads, size=k, replace=False)
        mask = np.zeros(num_leads, dtype=np.float32)
        mask[kept_leads] = 1.0
        return signal * mask

    def _gaussian_noise(self, signal):
        """Add per-lead Gaussian noise scaled to each lead's std."""
        per_lead_std = signal.std(axis=0, keepdims=True)
        noise = np.random.normal(
            0.0, RARE_AUG_NOISE_STD, size=signal.shape
        ).astype(signal.dtype) * per_lead_std
        return signal + noise

    def _time_shift(self, signal):
        """Circularly shift along time axis by a small random offset."""
        max_shift = int(signal.shape[0] * RARE_AUG_MAX_SHIFT_FRAC)
        if max_shift <= 0:
            return signal
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return signal
        return np.roll(signal, shift, axis=0)

    def _augment_rare(self, signal):
        return self._time_shift(self._gaussian_noise(signal))

    def __data_generation(self, batch_indices):
        """Loads and preprocesses batch data."""
        batch_signals = []
        batch_labels = []

        # Initialize SignalProcessor
        signal_processor = SignalProcessor(sampling_rate=self.sampling_rate, target_length=self.target_length, lead_mode=self.lead_mode, desired_lead=self.desired_lead)

        for idx in batch_indices:
            file_path = self.df.iloc[idx]['file_path']

            # Load raw signal using RawSignalLoader
            raw_loader = RawSignalLoader(file_path=file_path, lead_mode=self.lead_mode, desired_lead=self.desired_lead)
            raw_signals, _ = raw_loader.load_signal()

            # Process the raw signal using SignalProcessor
            processed_signals = signal_processor.process_signal(raw_signals)

            # Get label
            label = self.df.iloc[idx]['classes']
            encoded_label = self.label_encoder.transform([label])[0]

            if self.augment:
                if (
                    RARE_CLASS_AUG
                    and encoded_label in self.rare_class_indices
                ):
                    processed_signals = self._augment_rare(processed_signals)
                processed_signals = self._random_lead_mask(processed_signals)

            batch_signals.append(processed_signals)
            batch_labels.append(encoded_label)

        return (
            np.array(batch_signals, dtype=np.float32),
            np.array(batch_labels, dtype=np.int32)
        )

    def on_epoch_end(self):
        """Shuffles indexes after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)
