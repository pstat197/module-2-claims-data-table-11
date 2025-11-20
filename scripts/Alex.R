library(keras3)
library(tensorflow)
library(dplyr)

load("data/claims-clean-example.RData")

set.seed(187)
split_idx <- sample(seq_len(nrow(claims_clean)), size = 0.8 * nrow(claims_clean))
train_df <- claims_clean[split_idx, ]
test_df  <- claims_clean[-split_idx, ]

train_text <- train_df$text_clean
train_labels_bin <- as.numeric(train_df$bclass) - 1
train_labels_multi <- as.numeric(train_df$mclass) - 1
keep_idx <- !is.na(train_text) & !is.na(train_labels_bin) & !is.na(train_labels_multi)
train_text <- train_text[keep_idx]
train_labels_bin <- train_labels_bin[keep_idx]
train_labels_multi <- train_labels_multi[keep_idx] 

text_layer <- layer_text_vectorization(
  max_tokens = 20000,
  output_mode = "int",
  output_sequence_length = 200
)
text_layer |> adapt(train_text)

binary_model <- keras_model_sequential(
  layers = list(
    text_layer,
    layer_embedding(input_dim = 20000, output_dim = 64, mask_zero = TRUE),
    layer_bidirectional(layer = layer_lstm(units = 32)),
    layer_dense(units = 32, activation = "relu"),
    layer_dense(units = 1, activation = "sigmoid")
  )
)
binary_model |> compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

binary_model |> fit(
  x = train_text,
  y = train_labels_bin,
  batch_size = 32,
  epochs = 5,
  validation_split = 0.2
)

text_layer_multi <- layer_text_vectorization(
  max_tokens = 20000,
  output_mode = "int",
  output_sequence_length = 200
)
text_layer_multi |> adapt(train_text)

multi_model <- keras_model_sequential(
  layers = list(
    text_layer,
    layer_embedding(input_dim = 20000, output_dim = 64, mask_zero = TRUE),
    layer_bidirectional(layer = layer_lstm(units = 32)),
    layer_dense(units = 32, activation = "relu"),
    layer_dense(units = 5, activation = "softmax")
  )
)

multi_model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

multi_model |> fit(
  x = train_text,
  y = train_labels_multi,
  batch_size = 32,
  epochs = 8,
  validation_split = 0.2
)

load("data/claims-test.RData")

test_text <- test_df$text_clean
keep_idx_test <- !is.na(test_text)
test_text <- test_text[keep_idx_test]

binary_probs <- predict(binary_model, test_text)
binary_pred <- ifelse(binary_probs > 0.5, 1, 0)

multi_probs <- predict(multi_model, test_text)
multi_pred <- apply(multi_probs, 1, which.max) -1

pred_df <- tibble(
  .id = test_df$.id[keep_idx_test],
  bclass.pred = binary_pred,
  mclass.pred = multi_pred
)

pred_df

dir.create("results", showWarnings = FALSE)
saveRDS(binary_model, "results/binary_model.rds")
saveRDS(multi_model, "results/multi_model.rds")
saveRDS(pred_df, "results/pred_df.rds")

test_labels_bin  <- as.numeric(test_df$bclass) - 1
test_labels_multi <- as.numeric(test_df$mclass) - 1

test_labels_bin  <- test_labels_bin[keep_idx_test]
test_labels_multi <- test_labels_multi[keep_idx_test]

binary_accuracy <- mean(binary_pred == test_labels_bin)
multi_accuracy <- sum(multi_pred == test_labels_multi) / length(test_labels_multi)

binary_accuracy
multi_accuracy

library(yardstick)

binary_df <- tibble(
  truth = factor(test_labels_bin),
  estimate = factor(binary_pred)
)

binary_sens  <- sens_vec(binary_df$truth, binary_df$estimate, event_level = "second")
binary_spec  <- spec_vec(binary_df$truth, binary_df$estimate, event_level = "second")
binary_acc   <- mean(binary_df$truth == binary_df$estimate)

binary_results <- tibble(
  class = c(1,0),
  sensitivity = c(binary_sens, NA),
  specificity = c(binary_spec, NA),
  accuracy = binary_acc
)

binary_results

classes <- sort(unique(test_labels_multi))
results <- tibble(class = integer(), sensitivity = double(), specificity = double())

for (cl in classes) {
  truth_bin <- factor(ifelse(test_labels_multi == cl, cl, paste0("not_", cl)))
  pred_bin  <- factor(ifelse(multi_pred == cl, cl, paste0("not_", cl)))
  
  sens_val <- sens_vec(truth_bin, pred_bin, event_level = "first")
  spec_val <- spec_vec(truth_bin, pred_bin, event_level = "first")
  
  results <- bind_rows(results, tibble(class = cl, sensitivity = sens_val, specificity = spec_val))
}

accuracy_val <- mean(multi_pred == test_labels_multi)

results <- results %>% mutate(accuracy = accuracy_val)
results

multi_results <- results %>% select(class, sensitivity, specificity) %>%
  mutate(model = "multiclass")

binary_results_clean <- binary_results %>%
  filter(class == 1) %>% 
  select(class, sensitivity, specificity) %>%
  mutate(model = "binary")

combined_results <- bind_rows(binary_results_clean, multi_results) %>%
  arrange(model, class) %>%
  mutate(accuracy = c(binary_acc, rep(multi_accuracy, nrow(multi_results))))

combined_results

dir.create("results", showWarnings = FALSE)
write.csv(combined_results, "results/combined_metrics.csv", row.names = FALSE)
