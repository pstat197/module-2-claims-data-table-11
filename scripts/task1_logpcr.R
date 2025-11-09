# Task 1

need <- c("tidymodels","textrecipes","glmnet","xml2","rvest",
          "dplyr","stringr","purrr","tibble","yardstick","broom",
          "kableExtra","doParallel")
has  <- rownames(installed.packages())
if (!"rlang" %in% has || utils::packageVersion("rlang") < "1.1.6") install.packages("rlang")

to_install <- setdiff(need, has)

if (length(to_install)) install.packages(to_install, dependencies = TRUE)

suppressPackageStartupMessages({
  library(tidymodels); library(textrecipes); library(glmnet)
  library(xml2); library(rvest)
  library(dplyr); library(stringr); library(purrr); library(tibble)
  library(yardstick); library(broom); library(kableExtra); library(doParallel)
})

set.seed(197)
dir.create("results", showWarnings = FALSE)
dir.create("writeups", showWarnings = FALSE)


CV_FOLDS    <- 5
MAX_TOKENS  <- 30000
NUM_PCS     <- 300
GRID_LEN    <- 20

# Read-in the data & Function start-ups
load_first_df <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
  objs <- load(path)
  df_names <- objs[sapply(mget(objs, envir = parent.frame()), inherits, "data.frame")]
  if (!length(df_names)) stop("No data.frame objects inside: ", path)
  get(df_names[1], envir = parent.frame())
}

pick_by_names <- function(df, choices) choices[choices %in% names(df)][1]

detect_schema <- function(df) {
  html_col <- pick_by_names(df, c("html","raw_html","page_html","content","raw","body","text"))
  if (is.na(html_col)) {
    char_cols <- names(df)[sapply(df, inherits, "character")]
    looks_like_html <- function(v) any(str_detect(v, "(?i)<html|</p>|<p>|<h[1-6]|<!doctype"), na.rm = TRUE)
    html_col <- char_cols[sapply(char_cols, function(cn) looks_like_html(df[[cn]]))][1]
  }
  if (is.na(html_col)) stop("Could not detect an HTML column.")
  
  id_col <- pick_by_names(df, c(".id","id","url_id","page_id"))
  if (is.na(id_col)) {
    cand <- names(df)[sapply(df, function(x) dplyr::n_distinct(x, na.rm=TRUE) > 0.95*nrow(df))][1]
    if (is.na(cand)) stop("Could not detect an ID column.")
    id_col <- cand
  }
  
  b_col <- pick_by_names(df, c("bclass","binary_class","binary"))
  derived <- FALSE
  if (is.na(b_col)) {
    m_col <- pick_by_names(df, c("mclass","multiclass","class","label"))
    if (is.na(m_col)) stop("No class labels found (bclass/mclass/class/label).")
    b_col <- m_col; derived <- TRUE
  }
  list(html_col = html_col, id_col = id_col, b_col = b_col, derived = derived)
}

train_path <- "data/claims-raw.RData"
raw_df <- load_first_df(train_path)
sch <- detect_schema(raw_df)

if (sch$derived) {
  raw_df <- raw_df %>%
    mutate(bclass = factor(if_else(.data[[sch$b_col]] %in% c("claims","Claim","CLAIMS"),
                                   "claims","other")))
  b_col <- "bclass"
} else {
  b_col <- sch$b_col
}

claims_raw <- raw_df %>%
  transmute(
    .id   = .data[[sch$id_col]],
    html  = .data[[sch$html_col]],
    bclass = factor(.data[[b_col]])
  )

message(sprintf("Detected -> ID: %s | HTML: %s | bclass: %s", sch$id_col, sch$html_col, b_col))

extract_paragraphs <- function(html) {
  if (is.na(html) || !nzchar(html)) return("")
  doc <- tryCatch(read_html(html), error = function(e) NULL); if (is.null(doc)) return("")
  doc %>% html_elements("p") %>% html_text2() %>% str_squish() %>% str_c(collapse = " ")
}
extract_paragraphs_headers <- function(html) {
  if (is.na(html) || !nzchar(html)) return("")
  doc <- tryCatch(read_html(html), error = function(e) NULL); if (is.null(doc)) return("")
  p <- doc %>% html_elements("p") %>% html_text2()
  h <- doc %>% html_elements("h1,h2,h3,h4,h5,h6") %>% html_text2()
  c(h, p) %>% str_squish() %>% str_c(collapse = " ")
}

claims_para <- claims_raw %>%
  mutate(text = map_chr(html, extract_paragraphs)) %>%
  filter(nchar(text) > 0)

claims_hdrp <- claims_raw %>%
  mutate(text = map_chr(html, extract_paragraphs_headers)) %>%
  filter(nchar(text) > 0)

if (!foreach::getDoParRegistered()) {
  cl <- parallel::makeCluster(max(1, parallel::detectCores() - 1))
  doParallel::registerDoParallel(cl)
  on.exit({ try(parallel::stopCluster(cl), silent = TRUE) }, add = TRUE)
}

make_logpcr_wf <- function(dat, num_pca = NUM_PCS, max_tokens = MAX_TOKENS) {
  recipe(bclass ~ text, data = dat) %>%
    step_tokenize(text) %>%
    step_stopwords(text) %>%
    step_tokenfilter(text, max_tokens = max_tokens) %>%
    step_tfidf(text) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_predictors()) %>%
    step_pca(all_predictors(), num_comp = num_pca) %>%
    workflows::workflow() %>%
    workflows::add_model( logistic_reg(penalty = tune(), mixture = 1) %>% set_engine("glmnet") )
}

evaluate_logpcr <- function(dat, label,
                            v = CV_FOLDS, num_pca = NUM_PCS,
                            max_tokens = MAX_TOKENS, grid_len = GRID_LEN) {
  dat <- dat %>% filter(!is.na(text), nchar(text) > 0) %>% droplevels()
  if (n_distinct(dat$bclass) < 2) stop("Need two bclass levels in '", label, "' data.")
  folds <- vfold_cv(dat, v = v, strata = bclass)
  wf    <- make_logpcr_wf(dat, num_pca = num_pca, max_tokens = max_tokens)
  grid  <- tibble(penalty = 10^seq(-5, 0, length.out = grid_len))
  ctrl  <- control_grid(parallel_over = "resamples", save_pred = FALSE, verbose = TRUE)
  
  res <- tune_grid(
    wf, resamples = folds, grid = grid,
    metrics = metric_set(accuracy, roc_auc),
    control = ctrl
  )
  best <- select_best(res, metric = "accuracy")
  wf_f <- finalize_workflow(wf, best)
  
  summ <- tibble(
    setting = label,
    metric  = c("accuracy","roc_auc"),
    mean    = c(
      collect_metrics(res) %>% filter(.metric=="accuracy", .config==best$.config) %>% pull(mean),
      collect_metrics(res) %>% filter(.metric=="roc_auc",  .config==best$.config) %>% pull(mean)
    ),
    std_err = c(
      collect_metrics(res) %>% filter(.metric=="accuracy", .config==best$.config) %>% pull(std_err),
      collect_metrics(res) %>% filter(.metric=="roc_auc",  .config==best$.config) %>% pull(std_err)
    )
  )
  
  list(summary = summ, workflow = wf_f)
}

safe_eval <- function(dat, label) {
  tryCatch(evaluate_logpcr(dat, label),
           error = function(e) { warning(sprintf("'%s' skipped: %s", label, e$message)); NULL })
}

cat("\nFitting logistic PCR models…\n")
res_para <- safe_eval(claims_para, "paragraphs_only")
res_hdrp <- safe_eval(claims_hdrp, "headers_plus_paragraphs")

# Results
summaries <- list(res_para, res_hdrp) %>%
  compact() %>% map("summary") %>% bind_rows()

if (nrow(summaries)) {
  tbl <- summaries %>%
    arrange(metric, desc(mean)) %>%
    mutate(mean = round(mean, 4), std_err = round(std_err, 4)) %>%
    kbl(caption = sprintf(
      "Logistic PCR (%dfold CV, max_tokens=%d, PCs=%d, grid=%d)",
      CV_FOLDS, MAX_TOKENS, NUM_PCS, GRID_LEN
    ), align = "lccc") %>%
    kable_classic(full_width = FALSE)
  print(tbl)
  saveRDS(summaries, file = "results/task1-logpcr-summaries.rds")
  if (!is.null(res_para$workflow)) saveRDS(res_para$workflow, "results/task1-wf-paragraphs.rds")
  if (!is.null(res_hdrp$workflow)) saveRDS(res_hdrp$workflow, "results/task1-wf-headers_plus_paragraphs.rds")
} else {
  message("No results to display.")
}

