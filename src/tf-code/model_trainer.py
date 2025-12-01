import tensorflow as tf

class ModelTrainer:
    def __init__(self, max_epochs=20, patience=2):
        self.max_epochs = max_epochs
        self.patience = patience

    def compile_and_fit(self, model, window):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            mode='min'
        )
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        history = model.fit(
            window.train,
            epochs=self.max_epochs,
            validation_data=window.val,
            callbacks=[early_stopping]
        )
        return history