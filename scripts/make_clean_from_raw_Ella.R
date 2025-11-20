library(textclean)

source('scripts/preprocessing.R')
load('data/claims-raw.RData')
claims_clean <- claims_raw %>% 
  parse_data()

claims_clean <- claims_clean %>%
  mutate(
    bclass = case_when(
      str_detect(bclass, "Relevant claim content") ~ 1,
      str_detect(bclass, "No relevant content")   ~ 0,
      TRUE ~ NA_real_
    )
  )

save(claims_clean, file = 'data/claims-clean.RData')
