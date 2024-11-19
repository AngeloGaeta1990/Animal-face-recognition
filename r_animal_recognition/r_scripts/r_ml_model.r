library(keras)
library(tensorflow)

# Define image data generators for training and validation datasets
train_datagen <- image_data_generator(
  # Normalize pixel values to the 0-1 range
  rescale = 1 / 255,
  # Apply shear transformations with a range of 0.2 radians
  shear_range = 0.2,
  # Randomly zoom in/out by up to 20%
  zoom_range = 0.2,
  # Randomly flip images horizontally
  horizontal_flip = TRUE
)


# Only rescaling for test set
test_datagen <- image_data_generator(rescale = 1 / 255)

# Define directories containing training and validation images
train_dir <- "outputs/datasets/collection/train"
validation_dir <- "outputs/datasets/collection/val"

# Set image dimensions
img_width <- 150
img_height <- 150
input_shape <- c(img_width, img_height, 3)
num_classes <- 3

# Build the model using the Functional API
inputs <- layer_input(shape = input_shape)

# Generate batches of augmented data for training
train_generator <- flow_images_from_directory(
  directory = train_dir,
  generator = train_datagen,
  target_size = c(img_width, img_height),
  batch_size = 32,
  class_mode = "categorical"  # For multi-class classification
)

# Generate batches of data for validation (no augmentation)
validation_generator <- flow_images_from_directory(
  directory = validation_dir,
  generator = test_datagen,
  target_size = c(img_width, img_height),
  batch_size = 32,
  class_mode = "categorical"
)

outputs <- inputs %>%
  # First convolutional block
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%

  # Second convolutional block
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%

  # Third convolutional block
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%

  # Flatten and add dense layers
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%  # Dropout for regularization

  # Output layer
  layer_dense(units = num_classes, activation = "softmax")

# Define the model
model <- keras_model(inputs = inputs, outputs = outputs)

# Display the model's architecture
summary(model)

model %>% compile(
  loss = "categorical_crossentropy",  # For multi-class classification
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)

# Compute steps per epoch
train_steps <- ceiling(train_generator$samples / train_generator$batch_size)
validation_steps <- ceiling(validation_generator$samples / validation_generator$batch_size)


# Set the number of epochs
epochs <- 30

early_stopping <- callback_early_stopping(
  monitor = "val_loss",
  # Number of epochs with no improvement after which training will be stopped
  patience = 5,
  # Restore model weights from the epoch with the best value
  # of the monitored quantity
  restore_best_weights = TRUE
)

checkpoint <- callback_model_checkpoint(
  filepath = "best_animal_recognition_model.h5",
  monitor = "val_accuracy",
  save_best_only = TRUE
)

reduce_lr <- callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  # Factor by which the learning rate will be reduced
  factor = 0.1,
  # Number of epochs with no improvement after which
  # learning rate will be reduced
  patience = 3,
  min_lr = 1e-6
)

# Fit the model
history <- model %>% fit(
  train_generator,
  steps_per_epoch = train_steps,
  epochs = epochs,
  validation_data = validation_generator,
  validation_steps = validation_steps,
  callbacks = list(early_stopping, checkpoint, reduce_lr)
)

# Evaluate on validation data
scores <- model %>% evaluate(
  validation_generator,
  steps = validation_steps
)

# Print accuracy
cat("Validation accuracy:", scores["accuracy"], "\n")

# Save the model in HDF5 format
save_model_hdf5(model, filepath = "animal_recognition_model.h5")


# Plot the training and validation accuracy and loss at each epoch
plot(history)


# Determine the actual number of epochs completed
actual_epochs <- length(history$metrics$accuracy)

library(ggplot2)

# Prepare data for plotting
df <- data.frame(
  epoch = rep(1:actual_epochs, times = 2),
  accuracy = c(history$metrics$accuracy, history$metrics$val_accuracy),
  loss = c(history$metrics$loss, history$metrics$val_loss),
  dataset = rep(c("training", "validation"), each = actual_epochs)
)

# Plot accuracy over epochs
ggplot(df, aes(x = epoch, y = accuracy, color = dataset)) +
  geom_line() +
  labs(title = "Training and Validation Accuracy", y = "Accuracy") +
  theme_minimal()

# Plot loss over epochs
ggplot(df, aes(x = epoch, y = loss, color = dataset)) +
  geom_line() +
  labs(title = "Training and Validation Loss", y = "Loss") +
  theme_minimal()


# Path to the new image
test_image_path <- "inputs/tests/test_dog.jpg"

# Preprocess the image
img <- image_load(test_image_path, target_size = c(img_width, img_height))
x <- image_to_array(img)
x <- array_reshape(x, c(1, dim(x)))  # Reshape for batch
x <- x / 255  # Rescale

# Predict the class probabilities
predictions <- model %>% predict(x)

# Get the index of the class with the highest probability
# [1, ] since predictions is a matrix
predicted_class_index <- which.max(predictions[1, ])

# Map the index to the class label
class_labels <- names(train_generator$class_indices)
predicted_label <- class_labels[predicted_class_index]

# Print the predicted label
cat("Predicted label:", predicted_label, "\n")
