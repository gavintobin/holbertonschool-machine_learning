#!/usr/bin/env python3
'''task 6'''
import tensorflow as tf
import numpy as np
import GPyOpt
import os

# Generate some random data for the example
np.random.seed(0)
X_train = np.random.rand(100, 1)
y_train = 2 * X_train + 1 + 0.1 * np.random.randn(100, 1)

# Define a simple regression model
def build_model(learning_rate, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Define the objective function for Bayesian optimization
def objective(params):
    learning_rate, batch_size = params[0]
    model = build_model(learning_rate, batch_size)

    # Create a unique checkpoint filename based on hyperparameters
    checkpoint_path = f"checkpoint_lr_{learning_rate}_bs_{batch_size}.h5"

    # Set up early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    # Set up model checkpoint to save the best model during training
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min'
    )

    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2,
                        batch_size=int(batch_size),
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=0)

    best_val_loss = min(history.history['val_loss'])

    # Remove the model checkpoint file if not the best
    if history.history['val_loss'][-1] != best_val_loss:
        os.remove(checkpoint_path)

    return best_val_loss

# Define the bounds and type for hyperparameters
bounds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0.001, 0.1)},
          {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)}]

# Initialize Bayesian optimization
optimizer = GPyOpt.methods.BayesianOptimization(f=objective, domain=bounds)

# Run Bayesian optimization for 30 iterations
optimizer.run_optimization(max_iter=30)

# Get the best hyperparameters and best objective value
best_hyperparams = optimizer.X[np.argmin(optimizer.Y)]
best_val_loss = min(optimizer.Y)

 with open('bayes_opt.txt', 'w') as report_file:
        report_file.write("Best Hyperparameters: {}\n".format(best_hyperparameters))
        report_file.write("Best Metric: {}\n".format(best_val_loss))
        report_file.write(optimizer.get_evaluations())
