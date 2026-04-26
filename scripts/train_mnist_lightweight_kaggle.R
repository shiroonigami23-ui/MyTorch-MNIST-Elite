# Lightweight MNIST training in R for Kaggle
# Produces reproducible artifacts for docs and HF showcase.

suppressPackageStartupMessages({
  library(keras)
  library(jsonlite)
})

set.seed(23)

message("Loading MNIST from keras dataset...")
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Lightweight subset for fast iteration
train_n <- 20000
test_n <- 5000
epochs_n <- 8
batch_n <- 128

x_train <- x_train[1:train_n,,,drop=FALSE]
y_train <- y_train[1:train_n]
x_test <- x_test[1:test_n,,,drop=FALSE]
y_test <- y_test[1:test_n]

# Normalize and flatten
x_train <- array_reshape(x_train / 255, c(train_n, 28 * 28))
x_test <- array_reshape(x_test / 255, c(test_n, 28 * 28))

y_train_cat <- to_categorical(y_train, num_classes = 10)
y_test_cat <- to_categorical(y_test, num_classes = 10)

message("Building lightweight MLP model...")
model <- keras_model_sequential() |>
  layer_dense(units = 256, activation = "relu", input_shape = c(28 * 28)) |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = 128, activation = "relu") |>
  layer_dense(units = 10, activation = "softmax")

model |>
  compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )

message("Training...")
history <- model |>
  fit(
    x_train,
    y_train_cat,
    epochs = epochs_n,
    batch_size = batch_n,
    validation_split = 0.1,
    verbose = 2
  )

score <- model |>
  evaluate(x_test, y_test_cat, verbose = 0)

# Predictions for confusion matrix
pred_probs <- model |> predict(x_test, verbose = 0)
pred_labels <- max.col(pred_probs) - 1L
cm <- table(True = y_test, Pred = pred_labels)

dir.create("checkpoints", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
dir.create("visuals", showWarnings = FALSE, recursive = TRUE)

model_path <- "checkpoints/mnist_lightweight_mlp.rds"
saveRDS(model, file = model_path)

metrics <- list(
  test_loss = unname(score[[1]]),
  test_accuracy = unname(score[[2]]),
  epochs = epochs_n,
  batch_size = batch_n,
  train_samples = train_n,
  test_samples = test_n
)

history_df <- data.frame(
  epoch = seq_len(epochs_n),
  loss = unlist(history$metrics$loss),
  accuracy = unlist(history$metrics$accuracy),
  val_loss = unlist(history$metrics$val_loss),
  val_accuracy = unlist(history$metrics$val_accuracy)
)

write_json(metrics, "outputs/metrics.json", auto_unbox = TRUE, pretty = TRUE)
write.csv(history_df, "outputs/history.csv", row.names = FALSE)
write.csv(as.data.frame.matrix(cm), "outputs/confusion_matrix.csv", row.names = TRUE)

# Save a simple training curve image
png("visuals/r_training_curve.png", width = 900, height = 540)
plot(history_df$epoch, history_df$accuracy,
     type = "l", lwd = 3, col = "blue",
     ylim = c(0, 1), xlab = "Epoch", ylab = "Accuracy",
     main = "MyTorch Lightweight R Training Curve")
lines(history_df$epoch, history_df$val_accuracy, lwd = 3, col = "darkgreen")
legend("bottomright", legend = c("Train", "Validation"), col = c("blue", "darkgreen"), lwd = 3)
grid()
dev.off()

message(sprintf("Done. Test accuracy: %.4f", metrics$test_accuracy))
message("Saved model to checkpoints/mnist_lightweight_mlp.rds")
message("Saved metrics/history/confusion matrix to outputs/")
message("Saved training curve to visuals/r_training_curve.png")


