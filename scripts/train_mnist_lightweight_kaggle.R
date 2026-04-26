# Efficient MNIST training in R for Kaggle
# Goal: higher accuracy with a lightweight CNN + stable optimization.

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

# Full dataset for strong benchmark while keeping model lightweight.
train_n <- 60000
test_n <- 10000
epochs_n <- 12
batch_n <- 128

x_train <- x_train[1:train_n,,,drop=FALSE]
y_train <- y_train[1:train_n]
x_test <- x_test[1:test_n,,,drop=FALSE]
y_test <- y_test[1:test_n]

# Normalize and keep image structure for CNN: [N, 28, 28, 1]
x_train <- array_reshape(x_train / 255, c(train_n, 28, 28, 1))
x_test <- array_reshape(x_test / 255, c(test_n, 28, 28, 1))

message("Building lightweight CNN model...")
model <- keras_model_sequential() |>
  layer_conv_2d(filters = 24, kernel_size = c(3, 3), padding = "same", activation = "relu", input_shape = c(28, 28, 1)) |>
  layer_batch_normalization() |>
  layer_separable_conv_2d(filters = 48, kernel_size = c(3, 3), padding = "same", activation = "relu") |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_dropout(rate = 0.15) |>
  layer_separable_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same", activation = "relu") |>
  layer_max_pooling_2d(pool_size = c(2, 2)) |>
  layer_dropout(rate = 0.20) |>
  layer_flatten() |>
  layer_dense(units = 96, activation = "relu") |>
  layer_dropout(rate = 0.25) |>
  layer_dense(units = 10, activation = "softmax")

model |>
  compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "sparse_categorical_crossentropy",
    metrics = c("accuracy")
  )

callbacks <- list(
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 2, min_lr = 1e-5, verbose = 1),
  callback_early_stopping(monitor = "val_loss", patience = 4, restore_best_weights = TRUE, verbose = 1)
)

message("Training...")
history <- model |>
  fit(
    x_train,
    y_train,
    epochs = epochs_n,
    batch_size = batch_n,
    validation_split = 0.1,
    callbacks = callbacks,
    verbose = 2
  )

score <- model |> evaluate(x_test, y_test, verbose = 0)

# Predictions for confusion matrix
pred_probs <- model |> predict(x_test, verbose = 0)
pred_labels <- max.col(pred_probs) - 1L
cm <- table(True = y_test, Pred = pred_labels)

dir.create("checkpoints", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
dir.create("visuals", showWarnings = FALSE, recursive = TRUE)

model_path <- "checkpoints/mnist_lightweight_cnn_v2.rds"
saveRDS(model, file = model_path)

acc_key <- if ("accuracy" %in% names(history$metrics)) "accuracy" else names(history$metrics)[grepl("acc", names(history$metrics))][1]
val_acc_key <- if ("val_accuracy" %in% names(history$metrics)) "val_accuracy" else names(history$metrics)[grepl("val_.*acc", names(history$metrics))][1]

history_df <- data.frame(
  epoch = seq_along(unlist(history$metrics$loss)),
  loss = unlist(history$metrics$loss),
  accuracy = unlist(history$metrics[[acc_key]]),
  val_loss = unlist(history$metrics$val_loss),
  val_accuracy = unlist(history$metrics[[val_acc_key]])
)

metrics <- list(
  test_loss = unname(score[[1]]),
  test_accuracy = unname(score[[2]]),
  best_val_accuracy = max(history_df$val_accuracy),
  final_train_accuracy = tail(history_df$accuracy, 1),
  epochs_requested = epochs_n,
  epochs_ran = nrow(history_df),
  batch_size = batch_n,
  train_samples = train_n,
  test_samples = test_n,
  model_type = "lightweight_cnn_v2",
  params = as.numeric(count_params(model))
)

write_json(metrics, "outputs/metrics.json", auto_unbox = TRUE, pretty = TRUE)
write.csv(history_df, "outputs/history.csv", row.names = FALSE)
write.csv(as.data.frame.matrix(cm), "outputs/confusion_matrix.csv", row.names = TRUE)

# Save training curve image
png("visuals/r_training_curve.png", width = 1000, height = 560)
plot(history_df$epoch, history_df$accuracy,
     type = "l", lwd = 3, col = "#1f77b4",
     ylim = c(0.9, 1.0), xlab = "Epoch", ylab = "Accuracy",
     main = "MyTorch Efficient CNN - Accuracy Curve")
lines(history_df$epoch, history_df$val_accuracy, lwd = 3, col = "#2ca02c")
legend("bottomright", legend = c("Train", "Validation"), col = c("#1f77b4", "#2ca02c"), lwd = 3)
grid()
dev.off()

message(sprintf("Done. Test accuracy: %.4f", metrics$test_accuracy))
message(sprintf("Best val accuracy: %.4f", metrics$best_val_accuracy))
message(sprintf("Params: %d", as.integer(metrics$params)))
message("Saved model to checkpoints/mnist_lightweight_cnn_v2.rds")
