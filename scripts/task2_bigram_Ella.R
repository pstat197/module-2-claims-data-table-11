library(tidyverse)
library(tidytext)
library(Matrix)
library(rsample)
library(pROC)
library(workflows)
library(parsnip)
library(textrecipes)
library(tm)

load("data/claims-clean.RData")

#### Config

set.seed(197)                        # same seed as Task 1
CLEAN_PATH <- "data/claims-clean.RData"
WF_HDRP <- "results/task1-wf-headers_plus_paragraphs.rds"
WF_PARA <- "results/task1-wf-paragraphs.rds"
OUT_RDATA <- "results/task2_bigram_results.RData"
BASE_OOF   <- "results/task1_baseline_oof.RData"

K_PC <- 50
MIN_BIGRAM_FREQ <- 3

dir <- "results"
##### Load Data

load("data/claims-clean.RData")
dat <- claims_clean %>%
  transmute(id = .id, text = text_clean, y = as.integer(bclass)) %>%
  filter(!is.na(text), nchar(text) > 0, !is.na(y))
glimpse(dat)

#### Split
split <- initial_split(dat, prop = 0.8, strata = y)
tr <- training(split); va <- testing(split)

wf_path <- WF_HDRP
baseline_logit_valid <- NULL

wf <- readRDS(wf_path)

expected_lvls <- c("N/A: No relevant content.", "Relevant claim content")
tr_wf <- tr %>%
  mutate(bclass = factor(ifelse(y == 1,
                                "Relevant claim content",
                                "N/A: No relevant content."),
                         levels = expected_lvls)) %>%
  select(text, bclass)

va_wf <- va %>%
  mutate(bclass = factor(ifelse(y == 1,
                                "Relevant claim content",
                                "N/A: No relevant content."),
                         levels = expected_lvls)) %>%
  select(text, bclass)
wf_fit <- fit(wf, data = tr_wf)
preds  <- predict(wf_fit, new_data = va_wf, type = "prob")
prob   <- preds[[ grep("^\\.pred_", names(preds), value = TRUE)[1] ]]
prob   <- pmin(pmax(prob, 1e-6), 1 - 1e-6)

baseline_logit_valid <- qlogis(prob)
names(baseline_logit_valid) <- va$id
save(baseline_logit_valid, file = BASE_OOF)


tr_bi <- tr %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  count(id, bigram, name = "n")

keep_terms <- tr_bi %>%
  count(bigram, wt = n, name = "tot") %>%
  filter(tot >= MIN_BIGRAM_FREQ) %>%
  pull(bigram)

dtm_tr <- cast_dtm(tr_bi %>% filter(bigram %in% keep_terms),
                   document = id, term = bigram, value = n)
Xtr <- as.matrix(dtm_tr)
ytr <- tr$y[match(rownames(Xtr), tr$id)]

va_bi <- va %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  filter(bigram %in% keep_terms) %>%
  count(id, bigram, name = "n")
if (!all(va$id %in% va_bi$id) && length(keep_terms) > 0) {
  miss <- setdiff(va$id, unique(va_bi$id))
  if (length(miss)) {
    va_bi <- bind_rows(va_bi, tibble(id = miss[1], bigram = keep_terms[1], n = 0))
  }
}

dtm_va <- cast_dtm(va_bi, document = id, term = bigram, value = n)
Xva <- as.matrix(dtm_va)
yva <- va$y[match(rownames(Xva), va$id)]


missing_terms <- setdiff(colnames(Xtr), colnames(Xva))
if (length(missing_terms) > 0) {
  Xva <- cbind(
    Xva,
    matrix(
      0,
      nrow = nrow(Xva),
      ncol = length(missing_terms),
      dimnames = list(rownames(Xva), missing_terms)
    )
  )
}

extra_terms <- setdiff(colnames(Xva), colnames(Xtr))
if (length(extra_terms) > 0) {
  Xva <- Xva[, setdiff(colnames(Xva), extra_terms), drop = FALSE]
}

Xva <- Xva[, colnames(Xtr), drop = FALSE]

# ---- 5) PCA ----
stopifnot(nrow(Xtr) > 0, ncol(Xtr) > 0)
pca <- prcomp(Xtr, center = TRUE, scale. = TRUE)
k   <- min(K_PC, ncol(pca$x))
PCtr <- pca$x[, 1:k, drop = FALSE]
PCva <- scale(Xva, center = pca$center, scale = pca$scale) %*%
  pca$rotation[, 1:k, drop = FALSE]

# ---- 6) Logistic (bigram-only) ----
df_tr <- data.frame(y = factor(ytr), PCtr)
df_va <- data.frame(PCva)
fit_big <- glm(y ~ ., data = df_tr, family = binomial())
prob_big <- predict(fit_big, newdata = df_va, type = "response")
acc_big  <- mean((prob_big > 0.5) == yva)
auc_big  <- as.numeric(pROC::auc(yva, prob_big))
print(sprintf("Bigram-only   ACC=%.3f  AUC=%.3f  (k=%d, min_freq=%d)",
                acc_big, auc_big, k, MIN_BIGRAM_FREQ))

# ---- 7) Stacking: baseline logit + bigram PCs ----
fit_stack <- NULL; acc_stack <- NA_real_; auc_stack <- NA_real_

if (!exists("baseline_logit_valid") || is.null(baseline_logit_valid)) {
  if (file.exists(BASE_OOF)) {
    load(BASE_OOF)  # load baseline_logit_valid
    print("Loaded baseline_logit_valid from ", BASE_OOF)
  }
}

if (!is.null(baseline_logit_valid)) {

  base_aligned <- baseline_logit_valid[rownames(Xva)]
  
  ok <- !is.na(base_aligned)
  base_aligned <- base_aligned[ok]
  PCva_stack   <- PCva[ok, , drop = FALSE]
  yva_stack    <- yva[ok]
  
  if (length(base_aligned) == length(yva_stack)) {
    df_stack <- data.frame(
      y = factor(yva_stack),
      baseline_logit = as.numeric(base_aligned),
      PCva_stack
    )
    
    fit_stack  <- glm(y ~ ., data = df_stack, family = binomial())
    prob_stack <- predict(fit_stack, type = "response")
    acc_stack  <- mean((prob_stack > 0.5) == yva_stack)
    auc_stack  <- as.numeric(pROC::auc(yva_stack, prob_stack))
    print(sprintf("Stacked       ACC=%.3f  AUC=%.3f", acc_stack, auc_stack))
  } else {
    print("Stacking skipped (still mismatch after alignment).")
  }
} else {
  print("Stacking skipped (no baseline logits available).")
}

# ---- 8) Save ----
OUT_CSV <- "results/task2_metrics.csv"

save(pca, k, MIN_BIGRAM_FREQ,
     fit_big, acc_big, auc_big,
     fit_stack, acc_stack, auc_stack,
     file = OUT_RDATA)

tibble(
  model = c("bigram_only","stacked"),
  acc   = c(acc_big, acc_stack),
  auc   = c(auc_big, auc_stack),
  k_pc  = k,
  min_bigram_freq = MIN_BIGRAM_FREQ
) |>
  write_csv(OUT_CSV)

Print("Saved:\n- ", OUT_RDATA, "\n- ", OUT_CSV)