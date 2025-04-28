#Implementation of backpropagation in R for the MNIST dataset using the Keras package
library(keras)
library(tensorflow)

# Load the MNIST dataset
mnist <- dataset_mnist()

x_train <- mnist$train$x / 255
y_train <- mnist$train$y
x_test <- mnist$test$x / 255
y_test <- mnist$test$y

# Define the model architecture
model <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

# Compile the model
model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)

# Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  validation_split = 0.2
)

# Evaluate the model on the test set
model %>% evaluate(x_test, y_test)

# Plot the accuracy and loss curves
plot(history)


#Implementation of backpropagation in R for the CIFAR-10 dataset using the Keras package
library(keras)
library(tensorflow)

# Load the CIFAR-10 dataset
cifar10 <- dataset_cifar10()

x_train <- cifar10$train$x / 255
y_train <- cifar10$train$y
x_test <- cifar10$test$x / 255
y_test <- cifar10$test$y

# Define the model architecture
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(32, 32, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

# Compile the model
model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)

# Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  validation_split = 0.2
)

# Evaluate the model on the test set
model %>% evaluate(x_test, y_test)

# Plot the accuracy and loss curves
plot(history)
