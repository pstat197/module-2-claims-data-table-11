setwd("/Users/alexmorifusa/module-2-claims-data-table-11")

install.packages(c("tidyverse","quanteda","glmnet","xgboost","caret","textrecipes","recipes","rsample","yardstick","R.utils","xml2","rvest"))

library(tidyverse)
library(rvest)
library(quanteda)
library(glmnet)
library(xgboost)
library(caret)
library(rsample)
library(yardstick)
library(R.utils)

source("scripts/make_clean_from_raw_Ella.R")
source("scripts/preprocessing.R")
source("scripts/nlp-model-development.R")
source("scripts/prediction.R")

safe_load <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
  env <- new.env()
  load(path, envir = env)
  objs <- ls(env)
  if (length(objs) == 0) stop("No objects found in ", path)
  prefer <- c("claims_raw","claims_clean","claims_test")
  for (p in prefer) if (p %in% objs) return(get(p, envir = env))
  get(objs[1], envir = env)
}

extract_text_from_html <- function(html_string) {
  if (is.na(html_string) || is.null(html_string) || nchar(html_string) == 0) return("")
  doc <- tryCatch(read_html(html_string), error = function(e) NULL)
  if (is.null(doc)) return("")
  headers <- doc %>% html_nodes(xpath = ".//h1|.//h2|.//h3|.//h4") %>% html_text2()
  paras   <- doc %>% html_nodes("p") %>% html_text2()
  paste(c(headers, paras), collapse = " ")
}

train_obj <- safe_load("data/claims-raw.RData")
test_obj  <- safe_load("data/claims-test.RData")

train <- as_tibble(train_obj)
test  <- as_tibble(test_obj)

if (!("html" %in% names(train))) {
  guess <- intersect(names(train), c("raw_html","html_raw","content","text","page_html","html"))
  if (length(guess) >= 1) {
    train <- train %>% rename(html = !!sym(guess[1]))
  } else stop("Could not find HTML column in training data. Columns: ", paste(names(train), collapse = ", "))
}
if (!("html" %in% names(test))) {
  guess <- intersect(names(test), c("raw_html","html_raw","content","text","page_html","html"))
  if (length(guess) >= 1) {
    test  <- test %>% rename(html = !!sym(guess[1]))
  } else stop("Could not find HTML column in test data.")
}

if (!(".id" %in% names(train))) {
  if ("id" %in% names(train)) train <- train %>% rename(.id = id) else train <- train %>% mutate(.id = row_number())
}
if (!(".id" %in% names(test))) {
  if ("id" %in% names(test)) test <- test %>% rename(.id = id) else test <- test %>% mutate(.id = row_number())
}

train$text_raw <- map_chr(train$html, extract_text_from_html)
test$text_raw  <- map_chr(test$html, extract_text_from_html)

clean_text_basic <- function(x) {
  x %>%
    str_replace_all("\n", " ") %>%
    str_replace_all("\\s+", " ") %>%
    str_trim()
}

train$text_raw <- clean_text_basic(train$text_raw)
test$text_raw  <- clean_text_basic(test$text_raw)

save(train, file = "data/train_with_text.RData")
save(test, file = "data/test_with_text.RData")

combined_texts <- bind_rows(
  train %>% select(.id, text_raw) %>% mutate(split = "train"),
  test  %>% select(.id, text_raw) %>% mutate(split = "test")
)

corpus_all <- quanteda::corpus(combined_texts, text_field = "text_raw")

toks <- quanteda::tokens(corpus_all, remove_punct = TRUE, remove_numbers = TRUE)
toks <- tokens_tolower(toks)
toks <- tokens_select(toks, pattern = stopwords("en"), selection = "remove", padding = FALSE)

dfm_uni <- dfm(toks)
dfm_uni_trim <- dfm_trim(dfm_uni, min_termfreq = 5)

toks_bigram <- tokens_ngrams(toks, n = 2)
dfm_bi <- dfm(toks_bigram)
dfm_bi_trim <- dfm_trim(dfm_bi, min_termfreq = 3)

dfm_uni_tfidf <- dfm_tfidf(dfm_uni_trim)
dfm_bi_tfidf  <- dfm_tfidf(dfm_bi_trim)

meta <- docvars(corpus_all)
train_idx <- which(meta$split == "train")
test_idx  <- which(meta$split == "test")

X_uni_train <- convert(dfm_uni_tfidf[train_idx, ], to = "matrix")
X_uni_test  <- convert(dfm_uni_tfidf[test_idx, ], to = "matrix")
X_bi_train  <- convert(dfm_bi_tfidf[train_idx, ], to = "matrix")
X_bi_test   <- convert(dfm_bi_tfidf[test_idx, ], to = "matrix")

rownames(X_uni_train) <- combined_texts$.id[train_idx]
rownames(X_uni_test)  <- combined_texts$.id[test_idx]
rownames(X_bi_train)  <- combined_texts$.id[train_idx]
rownames(X_bi_test)   <- combined_texts$.id[test_idx]

if (!("bclass" %in% names(train))) {
  guess <- intersect(names(train), c("bclass","binary","label","y"))
  if (length(guess) >= 1) train <- train %>% rename(bclass = !!sym(guess[1])) else stop("No binary label 'bclass' found.")
}
if (!("mclass" %in% names(train))) {
  if ("class" %in% names(train)) train <- train %>% rename(mclass = class)
}

y_bin <- as.integer(as.character(train$bclass))
if (any(is.na(y_bin))) {
  y_bin <- as.integer(as.factor(train$bclass)) - 1
}

if ("mclass" %in% names(train)) {
  y_multi <- as.factor(train$mclass)
} else {
  y_multi <- NULL
}

uni_pca <- prcomp(X_uni_train, center = TRUE, scale. = TRUE)
explained <- cumsum(uni_pca$sdev^2) / sum(uni_pca$sdev^2)
ncomp <- min(which(explained >= 0.95))
ncomp <- min(ncomp, 200)

PC_train <- uni_pca$x[, 1:ncomp, drop = FALSE]
PC_test <- predict(uni_pca, newdata = X_uni_test)[, 1:ncomp, drop = FALSE]

cv_pcr <- cv.glmnet(x = PC_train, y = y_bin, family = "binomial", alpha = 0.5, nfolds = 5)
pred_pcr_train_prob <- predict(cv_pcr, newx = PC_train, s = "lambda.min", type = "response")
pred_pcr_test_prob  <- predict(cv_pcr, newx = PC_test,  s = "lambda.min", type = "response")

pred_pcr_train <- as.numeric(pred_pcr_train_prob > 0.5)

logit <- function(p) log(p / (1 - p))
log_odds_train <- as.numeric(logit(as.numeric(pred_pcr_train_prob)))
log_odds_test  <- as.numeric(logit(as.numeric(pred_pcr_test_prob)))

bi_pca <- prcomp(X_bi_train, center = TRUE, scale. = TRUE)
explained_bi <- cumsum(bi_pca$sdev^2) / sum(bi_pca$sdev^2)
ncomp_bi <- min(which(explained_bi >= 0.9))
ncomp_bi <- min(ncomp_bi, 200)

PC_bi_train <- bi_pca$x[, 1:ncomp_bi, drop = FALSE]
PC_bi_test  <- predict(bi_pca, newdata = X_bi_test)[, 1:ncomp_bi, drop = FALSE]

stack_train <- cbind(log_odds = log_odds_train, PC_bi_train)
stack_test  <- cbind(log_odds = log_odds_test,  PC_bi_test)

cv_stack <- cv.glmnet(x = stack_train, y = y_bin, family = "binomial", alpha = 0.5, nfolds = 5)
pred_stack_train_prob <- predict(cv_stack, newx = stack_train, s = "lambda.min", type = "response")
pred_stack_test_prob  <- predict(cv_stack, newx = stack_test,  s = "lambda.min", type = "response")

library(Matrix)
X_uni_train_sparse <- as(Matrix(X_uni_train, sparse = TRUE), "dgCMatrix")
X_uni_test_sparse  <- as(Matrix(X_uni_test, sparse = TRUE), "dgCMatrix")

cv_glmnet_uni <- cv.glmnet(x = X_uni_train_sparse, y = y_bin, family = "binomial", alpha = 0.5, nfolds = 5)
pred_glmnet_test_prob <- predict(cv_glmnet_uni, newx = X_uni_test_sparse, s = "lambda.min", type = "response")

dtrain <- xgb.DMatrix(data = X_uni_train, label = y_bin)
dtest  <- xgb.DMatrix(data = X_uni_test)
params <- list(objective = "binary:logistic", eval_metric = "auc", max_depth = 6, eta = 0.05, subsample = 0.8)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 200, verbose = 0)
pred_xgb_test_prob <- predict(xgb_model, newdata = dtest)

ensemble_prob_test <- (as.numeric(pred_stack_test_prob) + as.numeric(pred_xgb_test_prob)) / 2
bclass_pred_test <- ifelse(ensemble_prob_test > 0.5, 1, 0)

mclass_pred_test <- rep(NA, nrow(X_uni_test))
if (!is.null(y_multi)) {
  y_multi_vec <- as.integer(as.factor(train$mclass)) - 1
  cv_glmnet_multi <- cv.glmnet(x = X_uni_train_sparse, y = y_multi_vec, family = "multinomial", type.multinomial = "ungrouped", nfolds = 5)
  pred_multi_test_prob <- predict(cv_glmnet_multi, newx = X_uni_test_sparse, s = "lambda.min", type = "response")
  pred_arr <- pred_multi_test_prob[,,1]
  pred_idx <- max.col(pred_arr, ties.method = "first")
  levels_multi <- levels(as.factor(train$mclass))
  mclass_pred_test <- levels_multi[pred_idx]
}

pred_df <- tibble(
  .id = as.character(rownames(X_uni_test)),
  bclass.pred = ifelse(is.na(bclass_pred_test), 0, as.integer(bclass_pred_test)),
  mclass.pred = if (all(is.na(mclass_pred_test))) NA_character_ else as.character(mclass_pred_test)
)

if (!dir.exists("results")) dir.create("results")
write_csv(pred_df, "results/pred_df.csv")
saveRDS(pred_df, "results/pred_df.RDS")

saveRDS(cv_pcr, file = "results/cv_pcr.rds")
saveRDS(cv_stack, file = "results/cv_stack.rds")
saveRDS(cv_glmnet_uni, file = "results/cv_glmnet_uni.rds")
xgb.save(xgb_model, "results/xgb_model.model")
saveRDS(uni_pca, "results/uni_pca.rds")
saveRDS(bi_pca, "results/bi_pca.rds")

message("Done. Predictions written to results/pred_df.csv and results/pred_df.RDS")
