library(keras3)
library(tensorflow)
library(dplyr)

load("data/claims-clean-example.RData")

set.seed(187)
split_idx <- sample(seq_len(nrow(claims_clean)), size = 0.8 * nrow(claims_clean))
train_df <- claims_clean[split_idx, ]
test_df  <- claims_clean[-split_idx, ]

train_text <- train_df$text_clean
train_labels <- as.numeric(train_df$bclass) - 1
keep_idx <- !is.na(train_text) & !is.na(train_labels)
train_text <- train_text[keep_idx]
train_labels <- train_labels[keep_idx]

preprocess_layer <- keras3::layer_text_vectorization(
  standardize = NULL,
  split = "whitespace",
  max_tokens = 20000,
  output_mode = "tf_idf"
)
preprocess_layer |> keras3::adapt(train_text)

text_layer <- layer_text_vectorization(
  max_tokens = 20000,
  output_mode = "tf_idf"
)

text_layer |> keras3::adapt(train_text)

model <- keras_model_sequential(
  layers = list(
    text_layer,
    layer_dropout(rate = 0.2),
    layer_dense(units = 64, activation = "relu"),
    layer_dropout(rate = 0.3),
    layer_dense(units = 32, activation = "relu"),
    layer_dense(units = 1, activation = "sigmoid")
  )
)

model |> compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = "binary_accuracy"
)

history <- model |> fit(
  x = train_text,
  y = train_labels,
  batch_size = 32,
  epochs = 10,
  validation_split = 0.2
)

test_text <- test_df$text_clean
keep_idx_test <- !is.na(test_text)
test_text <- test_text[keep_idx_test]

pred_probs <- predict(model, test_text)
pred_classes <- ifelse(pred_probs > 0.5, 1, 0)

pred_df <- tibble(
  .id = test_df$.id[keep_idx_test],
  bclass.pred = pred_classes
)

saveRDS(pred_df, "results/modelAlex.rds")
