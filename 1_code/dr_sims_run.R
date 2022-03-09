# if (dgp == "ks") {
# 
# } else if (dgp == "nw") {
#   X <- matrix(rnorm(n * p), n, p)
#   tau <- X[, 1] + log(1 + exp(X[, 2]))
#   p <- 1 / (1 + exp(X[, 2] + X[, 3]))
#   e <- rep(0.5, n)
#   S <- rbinom(n, 1, p)
#   A <- rbinom(n, 1, e)
#   m <- pmax(0, X[, 1] + X[, 2], X[, 3]) + pmax(0, X[, 4] + X[, 5])
# }

# d <- generate_data(1000, 4)

# system.time({
#   simulate_learners(
#     X = d$X,
#     S = d$S,
#     A = d$A,
#     Y = d$Y,
#     Z = d$Z,
#     tau = d$tau,
#     m = d$m,
#     e = d$e,
#     p = d$p,
#     fold = d$fold,
#     p.correct = FALSE,
#     m.correct = FALSE
#   )
# })
# 
# 
# system.time({
#   simulate_learners2(
#     X = d$X,
#     S = d$S,
#     A = d$A,
#     Y = d$Y,
#     Z = d$Z,
#     tau = d$tau,
#     m = d$m,
#     e = d$e,
#     p = d$p,
#     fold = d$fold,
#     p.correct = FALSE,
#     m.correct = FALSE,
#     n.workers = 1
#   )
# })


library(tidyverse)
library(future)
library(parallel)
library(furrr)
library(progressr)
library(glmnet, exclude = c("expand", "pack", "unpack"))
library(randomForest, exclude = c("combine", "margin"))
library(xgboost, exclude = c("slice"))
library(nnet)
library(kernlab, exclude = c("cross", "alpha"))
library(SuperLearner)
library(sl3)
library(delayed)
library(tictoc)

sim1 <- 
  expand_grid(
    sim = 1:500,
    n = c(250, 500, 1250, 2500, 5000),
    m.correct = c(TRUE, FALSE),
    p.correct = c(TRUE, FALSE)
  )

sim1_batch1 <- 
  filter(sim1, p.correct == TRUE | m.correct == TRUE)

sim1_batch2 <- 
  filter(sim1, p.correct == FALSE & m.correct == FALSE)

tic()
plan(multisession, workers = 14)
with_progress({
  p <- progressor(steps = nrow(sim1_batch1))
  
  sim1_batch1 <- 
    sim1_batch1 %>%
    mutate(
      mse = future_pmap(list(n, m.correct, p.correct), function(n, m.correct, p.correct) {
        p()
        d <- generate_data(n, 4)
        simulate_learners2(
          X = d$X,
          S = d$S,
          A = d$A,
          Y = d$Y,
          Z = d$Z,
          tau = d$tau,
          m = d$m,
          e = d$e,
          p = d$p,
          fold = d$fold,
          m.correct = m.correct,
          p.correct = p.correct,
          n.workers = 1
        )
      }, 
      .options = furrr_options(seed = 215235))
    )
})
toc()
plan(sequential)


tic()
plan(multisession, workers = 5)
with_progress({
  p <- progressor(steps = nrow(sim1_batch2))
  
  sim1_batch2 <- 
    sim1_batch2 %>%
    mutate(
      mse = future_pmap(list(n, m.correct, p.correct), function(n, m.correct, p.correct) {
        p()
        d <- generate_data(n, 4)
        simulate_learners2(
          X = d$X,
          S = d$S,
          A = d$A,
          Y = d$Y,
          Z = d$Z,
          tau = d$tau,
          m = d$m,
          e = d$e,
          p = d$p,
          fold = d$fold,
          m.correct = m.correct,
          p.correct = p.correct,
          n.workers = 3
        )
      }, 
      .options = furrr_options(seed = 13423))
    )
})
toc()

plan(sequential)

sim1 <- bind_rows(sim1_batch1, sim1_batch2) 

write_rds(sim1, "Dropbox/Research/Danaei Lab/4_code/generalizing-trial-results/sim1.rds")

sim1 <- read_rds("sim1.rds")
plot_sim <- 
  sim1 %>%
  unnest(c(mse)) 


plot_sim <-
  plot_sim %>%
  mutate(
    mse_reg = mse[, 1],
    mse_ipw = mse[, 2],
    mse_dr = mse[, 3],
    mse_oracle = mse[, 4],
    mse_reg.sl = mse[, 5],
    mse_ipw.sl = mse[, 6],
    mse_dr.sl = mse[, 7],
    mse_oracle.sl = mse[, 8],
    n = n * 2
  ) %>%
  select(sim, n, m.correct, p.correct, starts_with("mse_")) %>%
  pivot_longer(cols = starts_with("mse_")) %>%
  separate(name, c("name", "estimator"), sep = "_") %>%
  mutate(estimator = str_replace(estimator, "dr.sl", "dr.np")) %>%
  filter(!is.na(value)) %>%
  filter(estimator %in% c("dr", "dr.np", "ipw", "oracle", "reg")) %>%
  mutate(
    estimator = factor(
      x = estimator,
      levels = c("ipw", "reg", "dr", "dr.np", "oracle")
    ),
    dgp = case_when(
      p.correct == TRUE & m.correct == TRUE ~ "Both models correct",
      p.correct == TRUE & m.correct == FALSE ~ "Outcome model misspecified",
      p.correct == FALSE & m.correct == TRUE ~ "Participation model misspecified",
      p.correct == FALSE & m.correct == FALSE ~ "Both models misspecified"
    ),
    dgp = factor(dgp, levels = c("Both models correct",
                                 "Outcome model misspecified",
                                 "Participation model misspecified",
                                 "Both models misspecified"))
  )


p <- ggplot(plot_sim,
       aes(
         x = factor(n),
         y = value,
         fill = estimator,
         color = estimator
       )) +
  facet_wrap(~dgp, scales = "free") +
  geom_boxplot(outlier.shape = NA) +
  theme_minimal(base_family = "Palatino", base_size = 14) +
  theme(
    panel.grid = element_blank(),
    axis.line = element_line(),
    axis.ticks = element_line()
  ) + 
  labs(
    x = "n",
    y = "RMSE"
  ) + 
  scale_color_brewer(name = "", palette = "Set1") +
  scale_y_continuous(limits = quantile(plot_sim$value, c(0.1, 0.9))) +
  #scale_y_log10(limits = quantile(plot_sim$value, c(0.1, 0.9))) +
  scale_fill_manual(name = "", values = alpha(RColorBrewer::brewer.pal(8, name = "Set1"), 0.7))
ggsave(
  p,
  filename = "Dropbox/Research/Danaei Lab/4_code/generalizing-trial-results/plot.pdf",
  width = 11,
  height = 7,
  dpi = 800,
)



plot_sim <- 
  sim1 %>%
  unnest(c(mse)) 

plot_sim <-
  plot_sim %>%
  mutate(
    mse_reg = mse[, 1],
    mse_ipw = mse[, 2],
    mse_dr = mse[, 3],
    mse_oracle = mse[, 4],
    mse_reg.sl = mse[, 5],
    mse_ipw.sl = mse[, 6],
    mse_dr.sl = mse[, 7],
    mse_oracle.sl = mse[, 8],
    n = n * 2
  ) %>%
  select(sim, n, m.correct, p.correct, starts_with("mse_")) %>%
  pivot_longer(cols = starts_with("mse_")) %>%
  separate(name, c("name", "estimator"), sep = "_") %>%
  mutate(estimator = str_replace(estimator, ".sl", ".np")) %>%
  filter(!is.na(value)) %>%
  filter(estimator %in% c("dr.np", "ipw.np", "oracle.np", "reg.np")) %>%
  mutate(
    estimator = factor(
      x = estimator,
      levels = c("ipw.np", "reg.np", "dr.np", "oracle.np")
    ),
    dgp = case_when(
      p.correct == TRUE & m.correct == TRUE ~ "Both models correct",
      p.correct == TRUE & m.correct == FALSE ~ "Outcome model misspecified",
      p.correct == FALSE & m.correct == TRUE ~ "Participation model misspecified",
      p.correct == FALSE & m.correct == FALSE ~ "Both models misspecified"
    ),
    dgp = factor(dgp, levels = c("Both models correct",
                                 "Outcome model misspecified",
                                 "Participation model misspecified",
                                 "Both models misspecified"))
  )


p2 <- ggplot(plot_sim,
       aes(
         x = factor(n),
         y = value,
         fill = estimator,
         color = estimator
       )) +
  facet_wrap(~dgp, scales = "free") +
  geom_boxplot(outlier.shape = NA) +
  theme_minimal(base_family = "Palatino", base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.line = element_line(),
    axis.ticks = element_line()
  ) + 
  labs(
    x = "n",
    y = "RMSE"
  ) + 
  scale_y_continuous(limits = quantile(plot_sim$value, c(0.1, 0.9))) +
  scale_color_manual(name = "",
                    values = RColorBrewer::brewer.pal(8, name = "Set1")[c(1,2,4,5)]) +
  scale_fill_manual(name = "",
                    values = alpha(RColorBrewer::brewer.pal(8, name = "Set1")[c(1,2,4,5)], 0.7))

ggsave(
  p2,
  filename = "plot2.pdf",
  width = 5.5,
  height = 3.5,
  dpi = 800,
)

# ggplot(plot_sim,
#        aes(
#          x = factor(n),
#          y = value,
#          fill = estimator,
#          color = estimator
#        )) +
#   facet_wrap(!m.correct ~ !p.correct, scales = "free") +
#   geom_violin(outlier.shape = NA) +
#   theme_minimal(base_family = "Palatino", base_size = 12) +
#   theme(
#     panel.grid = element_blank(),
#     axis.line = element_line(),
#     axis.ticks = element_line()
#   ) + 
#   scale_color_brewer(palette = "Set2") +
#   scale_y_continuous(limits = quantile(plot_sim$value, c(0.1, 0.9))) +
#   #scale_y_log10(limits = quantile(plot_sim$value, c(0.1, 0.9))) +
#   scale_fill_manual(values = alpha(RColorBrewer::brewer.pal(8, name = "Set1"), 0.6))


                    