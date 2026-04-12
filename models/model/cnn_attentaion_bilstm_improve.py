import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam

class EnhancedCNNModel:
    """
    An enhanced 1D CNN model with correct attention mechanism implementation and various improvements.
    Each existing Conv1D layer is now followed by another Conv1D layer with similar parameters.
    """

    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):

        inputs = layers.Input(shape=self.input_shape)

        # First Conv Block
        x = layers.Conv1D(32, 15, strides=2, dilation_rate=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(32, 15, strides=1, dilation_rate=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001))(x) # Added Conv1D
        x = layers.BatchNormalization()(x) # Added BatchNormalization
        x = layers.Activation('relu')(x) # Added Activation
        x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
        x = layers.Dropout(0.2)(x)

        # Second Conv Block
        x = layers.Conv1D(64, 15, strides=2, dilation_rate=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(64, 15, strides=1, dilation_rate=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # Added Conv1D
        x = layers.BatchNormalization()(x) # Added BatchNormalization
        x = layers.Activation('relu')(x) # Added Activation
        x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
        x = layers.Dropout(0.3)(x)

        # Third Conv Block 
        x = layers.Conv1D(128, 15, strides=2, dilation_rate=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(128, 15, strides=1, dilation_rate=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # Added Conv1D
        x = layers.BatchNormalization()(x) # Added BatchNormalization
        x = layers.Activation('relu')(x) # Added Activation
        x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
        x = layers.Dropout(0.3)(x)
        
       
    
        # Attention Mechanism
        attention = layers.Conv1D(128, 1, strides=1, activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])  # Element-wise multiplication

        # Apply more pooling and dropout
        x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
        x = layers.Dropout(0.4)(x)

        # Bidirectional LSTM with stacked layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
       

        # Classifier
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        # Build the model
        model = models.Model(inputs=inputs, outputs=outputs, name="CNN_BiLSTM")

        # Compile the model with Adam optimizer and gradient clipping
        optimizer = Adam(learning_rate=self.learning_rate, clipvalue=1.0)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def get_model(self):
        return self.model

