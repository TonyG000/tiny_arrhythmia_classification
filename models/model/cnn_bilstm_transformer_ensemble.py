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


class EnsembleCNNBiLSTMTransformerModel:
    """
    Feature-level fusion ensemble: shared CNN backbone splits into parallel
    BiLSTM (local temporal features) and Transformer (global pattern) branches,
    whose outputs are concatenated before a dense fusion classifier.

    Inspired by: Jahangir R et al., BMC Cardiovasc Disord. 2025 Apr 7;25(1):260.
    """

    def __init__(self, input_shape, num_classes, learning_rate=0.001,
                 bilstm_units=(128, 64), transformer_heads=4,
                 transformer_ff_dim=256, dropout_fusion=0.5,
                 dense_fusion_units=256):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.bilstm_units = bilstm_units
        self.transformer_heads = transformer_heads
        self.transformer_ff_dim = transformer_ff_dim
        self.dropout_fusion = dropout_fusion
        self.dense_fusion_units = dense_fusion_units
        self.model = self.build_model()

    def _build_cnn_backbone(self, inputs):
        # First Conv Block
        x = layers.Conv1D(32, 15, strides=2, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(32, 15, strides=1, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
        x = layers.Dropout(0.2)(x)

        # Second Conv Block
        x = layers.Conv1D(64, 15, strides=2, padding='same',
                          kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(64, 15, strides=1, padding='same',
                          kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
        x = layers.Dropout(0.3)(x)

        # Third Conv Block
        x = layers.Conv1D(128, 15, strides=2, padding='same',
                          kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(128, 15, strides=1, padding='same',
                          kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
        x = layers.Dropout(0.3)(x)

        # Attention Gate
        attention = layers.Conv1D(128, 1, strides=1, activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])
        x = layers.MaxPooling1D(2, strides=2, padding='same', name='gradcam_target')(x)
        x = layers.Dropout(0.4)(x)

        return x

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        cnn_out = self._build_cnn_backbone(inputs)

        # --- BiLSTM Branch (local temporal features) ---
        bilstm = layers.Bidirectional(
            layers.LSTM(self.bilstm_units[0], return_sequences=True)
        )(cnn_out)
        bilstm = layers.Bidirectional(
            layers.LSTM(self.bilstm_units[1], return_sequences=True)
        )(bilstm)
        bilstm = layers.Dropout(0.3)(bilstm)
        bilstm_feat = layers.GlobalAveragePooling1D()(bilstm)

        # --- Transformer Branch (global pattern features) ---
        trans = TransformerEncoderBlock(
            embed_dim=128,
            num_heads=self.transformer_heads,
            ff_dim=self.transformer_ff_dim,
            dropout=0.1
        )(cnn_out)
        trans = TransformerEncoderBlock(
            embed_dim=128,
            num_heads=self.transformer_heads,
            ff_dim=self.transformer_ff_dim,
            dropout=0.1
        )(trans)
        trans = layers.Dropout(0.3)(trans)
        trans_feat = layers.GlobalAveragePooling1D()(trans)

        # --- Fusion Head ---
        merged = layers.Concatenate()([bilstm_feat, trans_feat])
        x = layers.Dropout(self.dropout_fusion)(merged)
        x = layers.Dense(self.dense_fusion_units, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(self.dropout_fusion)(x)
        x = layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs,
                             name="CNN_BiLSTM_Transformer_Ensemble")
        optimizer = Adam(learning_rate=self.learning_rate, clipvalue=1.0)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def get_model(self):
        return self.model
