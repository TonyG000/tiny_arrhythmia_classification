import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class CNNTransformerModel:
    """
    CNN-Attention-Transformer model: replaces BiLSTM layers from the baseline
    with a Transformer encoder block for improved long-range dependency modeling.
    """
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        # First Conv Block (identical to baseline)
        x = layers.Conv1D(32, 15, strides=2, dilation_rate=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(32, 15, strides=1, dilation_rate=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
        x = layers.Dropout(0.2)(x)

        # Second Conv Block (identical to baseline)
        x = layers.Conv1D(64, 15, strides=2, dilation_rate=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(64, 15, strides=1, dilation_rate=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
        x = layers.Dropout(0.3)(x)

        # Third Conv Block (identical to baseline)
        x = layers.Conv1D(128, 15, strides=2, dilation_rate=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(128, 15, strides=1, dilation_rate=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
        x = layers.Dropout(0.3)(x)

        # Attention Mechanism (identical to baseline)
        attention = layers.Conv1D(128, 1, strides=1, activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])
        x = layers.MaxPooling1D(2, strides=2, padding='same', name='gradcam_target')(x)
        x = layers.Dropout(0.4)(x)

        # Transformer Encoder (replaces BiLSTM)
        x = TransformerEncoderBlock(embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1)(x)
        x = TransformerEncoderBlock(embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1)(x)
        x = layers.Dropout(0.3)(x)

        # Classifier (identical to baseline)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="CNN_Transformer")
        optimizer = Adam(learning_rate=self.learning_rate, clipvalue=1.0)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def get_model(self):
        return self.model
