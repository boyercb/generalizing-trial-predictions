generate_data <- function(n, d, n.folds = 3, sigma.noise = 1) {
  fold <- rep(1:n.folds, each = n)
  n <- n * n.folds
  
  X <- matrix(rnorm(n * d), n, d)
  tau <- (10 + 13.7 * (2 * X[, 1] + X[, 2] + X[, 3] + X[, 4]))
  p <- plogis(-X[, 1] + 0.5 * X[, 2] - 0.25 * X[, 3] - 0.1 * X[, 4])
  e <- rep(0.5, n)
  S <- rbinom(n, 1, p)
  A <- rbinom(n, 1, e)
  m <- 200
  
  Z <- matrix(rnorm(n * d), n, d)
  Z[, 1] <- exp(X[, 1] / 2)
  Z[, 2] <- X[, 2] / (1 + exp(X[, 1])) + 10
  Z[, 3] <- (X[, 1] * X[, 3] / 25 + 0.6)^3
  Z[, 4] <- (X[, 2] + X[, 4] + 20)^2
  
  Y <- m + (A - 0.5) * tau + rnorm(n, 0, sigma.noise)
  
  out <-
    list(
      X = X,
      S = S,
      A = A,
      Y = Y,
      Z = Z,
      tau = tau,
      m = m,
      e = e,
      p = p,
      fold = fold
    )
  
  return(out)
}

simulate_learners <- function(
  X,
  S,
  A,
  Y,
  Z,
  tau,
  m,
  e,
  p,
  fold,
  a = 1,
  p.correct = TRUE,
  m.correct = TRUE,
  sl.lib = c(
    "SL.xgboost",
    "SL.randomForest",
    "SL.glmnet",
    "SL.nnet",
    "SL.gam",
    "SL.lm",
    "SL.mean"
  )
) {
  
  Xmat <- cbind(1, X)
  Zmat <- cbind(1, Z)
  
  Ya <- Y + (a - A) * tau
  EYa <- m + (a - 0.5) * tau
  
  n <- length(Y)
  n.folds <- length(unique(fold))
  
  if (p.correct) {
    fit_p <- glm.fit(
      x = Xmat[fold %in% c(1,2), ],
      y = S[fold %in% c(1,2)],
      family = binomial(link = "logit")
    )
    p.hat <- c(plogis(Xmat %*% coef(fit_p)))
  } else {
    fit_p <- glm.fit(
      x = Zmat[fold %in% c(1,2), ],
      y = S[fold %in% c(1,2)],
      family = binomial(link = "logit")
    )
    p.hat <- c(plogis(Zmat %*% coef(fit_p)))
  }
  
  fit_e <- glm.fit(
    x = Xmat[S == 1 & fold %in% c(1,2),],
    y = A[S == 1 & fold %in% c(1,2)],
    family = binomial(link = "logit")
  )
  
  e.hat <- c(plogis(Xmat %*% coef(fit_e)))
  e.hat <- a * e.hat + (1 - a) * (1 - e.hat)
  
  if (m.correct) {
    fit_m <- lm.fit(
      x = Xmat[A == a & S == 1 & fold %in% c(1,2),],
      y = Y[A == a & S == 1 & fold %in% c(1,2)]
    )
    m.hat <- c(Xmat %*% coef(fit_m))
  } else {
    fit_m <- lm.fit(
      x = Zmat[A == a & S == 1 & fold %in% c(1,2),],
      y = Y[A == a & S == 1 & fold %in% c(1,2)]
    )
    m.hat <- c(Zmat %*% coef(fit_m))
  }
  
  pseudo <- I(A == a) * S / (p.hat * e.hat) * Y
  inff <- (A == a) * S / (p.hat * e.hat) * (Y - m.hat) + m.hat
  
  fit_ipw <- lm.fit(x = Xmat, y = pseudo)
  fit_dr <- lm.fit(x = Xmat, y = inff)
  fit_oracle <- lm.fit(x = Xmat, y = Ya)
  
  stabwt <- sum(mean(I(A == a) * S / (p.hat * e.hat)))
  
  ipw <- c(Xmat %*% coef(fit_ipw)) / stabwt
  dr <- c(Xmat %*% coef(fit_dr)) / stabwt - m.hat / stabwt + m.hat
  oracle <- c(Xmat %*% coef(fit_oracle))
  
  if (!p.correct & !m.correct) {
    
    sl_p1 <- SuperLearner(
      X = data.frame(Z[fold == 1, ]),
      Y = S[fold == 1],
      SL.library = sl.lib,
      family = binomial()
    )
    sl_p2 <- SuperLearner(
      X = data.frame(Z[fold == 2, ]),
      Y = S[fold == 2],
      SL.library = sl.lib,
      family = binomial()
    )
    
    sl_e1 <- SuperLearner(
      X = data.frame(X[S == 1 & fold == 1, ]),
      Y = A[S == 1 & fold == 1],
      SL.library = sl.lib,
      family = binomial()
    )
    sl_e2 <- SuperLearner(
      X = data.frame(X[S == 1 & fold == 2, ]),
      Y = A[S == 1 & fold == 2],
      SL.library = sl.lib,
      family = binomial()
    )
    
    sl_m1 <- SuperLearner(
      X = data.frame(Z[A == a & S == 1 & fold == 1, ]),
      Y = Y[A == a & S == 1 & fold == 1],
      SL.library = sl.lib
    )
    sl_m2 <- SuperLearner(
      X = data.frame(Z[A == a & S == 1 & fold == 2, ]),
      Y = Y[A == a & S == 1 & fold == 2],
      SL.library = sl.lib
    )
    
    p.hat.sl1 <- predict(sl_p1, newdata = data.frame(Z), onlySL = TRUE)$pred
    e.hat.sl1 <- predict(sl_e1, newdata = data.frame(X), onlySL = TRUE)$pred
    m.hat.sl1 <- predict(sl_m1, newdata = data.frame(Z), onlySL = TRUE)$pred
    
    e.hat.sl1 <- a * e.hat.sl1 + (1 - a) * (1 - e.hat.sl1)
    
    pseudo.sl1 <- I(A == a) * S / (p.hat.sl1 * e.hat.sl1) * Y
    inff.sl1 <- (A == a) * S / (p.hat.sl1 * e.hat.sl1) * (Y - m.hat.sl1) + m.hat.sl1
    
    sl_ipw1 <- SuperLearner(
      X = data.frame(X[fold == 2, ]),
      Y = pseudo.sl1[fold == 2],
      SL.library = sl.lib
    )
    sl_dr1 <- SuperLearner(
      X = data.frame(X[fold == 2, ]),
      Y = inff.sl1[fold == 2],
      SL.library = sl.lib
    )
    sl_oracle1 <- SuperLearner(
      X = data.frame(X[fold == 2, ]),
      Y = Ya[fold == 2],
      SL.library = sl.lib
    )
    
    stabwt.sl1 <- sum(mean(I(A == a) * S / (p.hat.sl1 * e.hat.sl1)))
    
    ipw.sl1 <- predict(sl_ipw1, newdata = data.frame(X), onlySL = TRUE)$pred / stabwt.sl1
    dr.sl1 <- predict(sl_dr1, newdata = data.frame(X), onlySL = TRUE)$pred / stabwt.sl1 - m.hat.sl1 / stabwt.sl1 + m.hat.sl1
    oracle.sl1 <- predict(sl_oracle1, newdata = data.frame(X), onlySL = TRUE)$pred
    
    p.hat.sl2 <- predict(sl_p2, newdata = data.frame(X), onlySL = TRUE)$pred
    e.hat.sl2 <- predict(sl_e2, newdata = data.frame(X), onlySL = TRUE)$pred
    m.hat.sl2 <- predict(sl_m2, newdata = data.frame(X), onlySL = TRUE)$pred
    
    e.hat.sl2 <- a * e.hat.sl2 + (1 - a) * (1 - e.hat.sl2)
    
    pseudo.sl2 <- I(A == a) * S / (p.hat.sl2 * e.hat.sl2) * Y
    inff.sl2 <- (A == a) * S / (p.hat.sl2 * e.hat.sl2) * (Y - m.hat.sl2) + m.hat.sl2
    
    sl_ipw2 <- SuperLearner(
      X = data.frame(X[fold == 1, ]),
      Y = pseudo.sl2[fold == 1],
      SL.library = sl.lib
    )
    sl_dr2 <- SuperLearner(
      X = data.frame(X[fold == 1, ]),
      Y = inff.sl2[fold == 1],
      SL.library = sl.lib
    )
    sl_oracle2 <- SuperLearner(
      X = data.frame(X[fold == 1, ]),
      Y = Ya[fold == 1],
      SL.library = sl.lib
    )
    
    stabwt.sl2 <- sum(mean(I(A == a) * S / (p.hat.sl2 * e.hat.sl2)))
    
    ipw.sl2 <- predict(sl_ipw2, newdata = data.frame(X), onlySL = TRUE)$pred / stabwt.sl2
    dr.sl2 <- predict(sl_dr2, newdata = data.frame(X), onlySL = TRUE)$pred  / stabwt.sl2 - m.hat.sl2 / stabwt.sl2 + m.hat.sl2
    oracle.sl2 <- predict(sl_oracle2, newdata = data.frame(X), onlySL = TRUE)$pred
    
  } else {
    m.hat.sl1 <- NA
    m.hat.sl2 <- NA
    ipw.sl1 <- NA 
    ipw.sl2 <- NA
    dr.sl1 <- NA
    dr.sl2 <- NA
    oracle.sl1 <- NA
    oracle.sl2 <- NA
  }
  
  mses <-
    sapply(
      list(
        m.hat,
        ipw,
        dr,
        oracle,
        (m.hat.sl1 + m.hat.sl2) / 2,
        (ipw.sl1 + ipw.sl2) / 2,
        (dr.sl1 + dr.sl2) / 2,
        (oracle.sl1 + oracle.sl2) / 2
      ),
      function(x) sqrt(mean((EYa - x)[fold == 3] ^ 2))
    )
  
  mat <- matrix(
    data = c(mses),
    nrow = 1,
    ncol = 8,
    byrow = TRUE
  )
  
  return(mat)
}



simulate_learners2 <- function(
  X,
  S,
  A,
  Y,
  Z,
  tau,
  m,
  e,
  p,
  fold,
  a = 1,
  p.correct = TRUE,
  m.correct = TRUE,
  n.workers = 2
) {

  Xmat <- cbind(1, X)
  Zmat <- cbind(1, Z)

  Ya <- Y + (a - A) * tau
  EYa <- m + (a - 0.5) * tau

  n <- length(Y)
  n.folds <- length(unique(fold))

  if (p.correct) {
    fit_p <- glm.fit(
      x = Xmat[fold %in% c(1,2), ],
      y = S[fold %in% c(1,2)],
      family = binomial(link = "logit")
    )
    p.hat <- c(plogis(Xmat %*% coef(fit_p)))
  } else {
    fit_p <- glm.fit(
      x = Zmat[fold %in% c(1,2), ],
      y = S[fold %in% c(1,2)],
      family = binomial(link = "logit")
    )
    p.hat <- c(plogis(Zmat %*% coef(fit_p)))
  }

  fit_e <- glm.fit(
    x = Xmat[S == 1 & fold %in% c(1,2),],
    y = A[S == 1 & fold %in% c(1,2)],
    family = binomial(link = "logit")
  )
  e.hat <- c(plogis(Xmat %*% coef(fit_e)))
  e.hat <- a * e.hat + (1 - a) * (1 - e.hat)
  
  if (m.correct) {
    fit_m <- lm.fit(
      x = Xmat[A == a & S == 1 & fold %in% c(1,2),],
      y = Y[A == a & S == 1 & fold %in% c(1,2)]
    )
    m.hat <- c(Xmat %*% coef(fit_m))
  } else {
    fit_m <- lm.fit(
      x = Zmat[A == a & S == 1 & fold %in% c(1,2),],
      y = Y[A == a & S == 1 & fold %in% c(1,2)]
    )
    m.hat <- c(Zmat %*% coef(fit_m))
  }

  pseudo <- I(A == a) * S / (p.hat * e.hat) * Y
  inff <- (A == a) * S / (p.hat * e.hat) * (Y - m.hat) + m.hat

  fit_ipw <- lm.fit(x = Xmat, y = pseudo)
  fit_dr <- lm.fit(x = Xmat, y = inff)
  fit_oracle <- lm.fit(x = Xmat, y = Ya)

  stabwt <- sum(mean(I(A == a) * S / (p.hat * e.hat)))

  ipw <- c(Xmat %*% coef(fit_ipw)) / stabwt
  dr <- c(Xmat %*% coef(fit_dr)) / stabwt - m.hat / stabwt + m.hat
  oracle <- c(Xmat %*% coef(fit_oracle))

  if (!p.correct & !m.correct) {
    inc <- data.frame(cbind(Z, S, A, Y, Ya))
    cor <- data.frame(cbind(X, S, A, Y, Ya))
    colnames(inc) <- c("X1", "X2", "X3", "X4", "S", "A", "Y", "Ya")
    colnames(cor) <- c("X1", "X2", "X3", "X4", "S", "A", "Y", "Ya")

    p1_task <-
      sl3_Task$new(
        inc[fold == 1, ],
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "S",
        outcome_type = "binomial"
      )

    p2_task <-
      sl3_Task$new(
        inc[fold == 2, ],
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "S",
        outcome_type = "binomial"
      )

    e1_task <-
      sl3_Task$new(
        cor[fold == 1 & S == 1, ],
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "A",
        outcome_type = "binomial"
      )

    e2_task <-
      sl3_Task$new(
        cor[fold == 2 & S == 1, ],
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "A",
        outcome_type = "binomial"
      )

    m1_task <-
      sl3_Task$new(
        inc[fold == 1 & S == 1 & A == a, ],
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "Y",
        outcome_type = "continuous"
      )

    m2_task <-
      sl3_Task$new(
        inc[fold == 2 & S == 1 & A == a, ],
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "Y",
        outcome_type = "continuous"
      )

    stack <-
      Stack$new(
        Lrnr_glm$new(),
        Lrnr_mean$new(),
        Lrnr_gam$new(),
        Lrnr_randomForest$new(),
        Lrnr_xgboost$new(verbose = 0, eval_metric = 'logloss'),
        Lrnr_nnet$new(trace = FALSE)
      )

    stack_cont <-
      Stack$new(
        Lrnr_glm$new(),
        Lrnr_mean$new(),
        Lrnr_gam$new(),
        Lrnr_randomForest$new(),
        Lrnr_xgboost$new(verbose = 0)
      )
    
    sl <- Lrnr_sl$new(learners = stack)
    sl_cont <- Lrnr_sl$new(learners = stack_cont)
    
    p1_dsl <- delayed_learner_train(sl, p1_task)
    p2_dsl <- delayed_learner_train(sl, p2_task)
    e1_dsl <- delayed_learner_train(sl, e1_task)
    e2_dsl <- delayed_learner_train(sl, e2_task)
    m1_dsl <- delayed_learner_train(sl_cont, m1_task)
    m2_dsl <- delayed_learner_train(sl_cont, m2_task)

    plan(multisession, workers = n.workers)

    p1_sched <- Scheduler$new(p1_dsl, FutureJob, nworkers = n.workers)
    p2_sched <- Scheduler$new(p2_dsl, FutureJob, nworkers = n.workers)
    e1_sched <- Scheduler$new(e1_dsl, FutureJob, nworkers = n.workers)
    e2_sched <- Scheduler$new(e2_dsl, FutureJob, nworkers = n.workers)
    m1_sched <- Scheduler$new(m1_dsl, FutureJob, nworkers = n.workers)
    m2_sched <- Scheduler$new(m2_dsl, FutureJob, nworkers = n.workers)

    sl_p1 <- p1_sched$compute()
    sl_p2 <- p2_sched$compute()
    sl_e1 <- e1_sched$compute()
    sl_e2 <- e2_sched$compute()
    sl_m1 <- m1_sched$compute()
    sl_m2 <- m2_sched$compute()

    plan(sequential)

    p1_task <-
      sl3_Task$new(
        inc,
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "S",
        outcome_type = "binomial"
      )

    p2_task <-
      sl3_Task$new(
        inc,
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "S",
        outcome_type = "binomial"
      )

    e1_task <-
      sl3_Task$new(
        cor,
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "A",
        outcome_type = "binomial"
      )

    e2_task <-
      sl3_Task$new(
        cor,
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "A",
        outcome_type = "binomial"
      )

    m1_task <-
      sl3_Task$new(
        inc,
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "Y",
        outcome_type = "continuous"
      )

    m2_task <-
      sl3_Task$new(
        inc,
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "Y",
        outcome_type = "continuous"
      )

    p.hat.sl1 <- sl_p1$predict(p1_task)
    e.hat.sl1 <- sl_e1$predict(e1_task)
    m.hat.sl1 <- sl_m1$predict(m1_task)

    p.hat.sl2 <- sl_p2$predict(p2_task)
    e.hat.sl2 <- sl_e2$predict(e2_task)
    m.hat.sl2 <- sl_m2$predict(m2_task)

    e.hat.sl1 <- a * e.hat.sl1 + (1 - a) * (1 - e.hat.sl1)
    e.hat.sl2 <- a * e.hat.sl2 + (1 - a) * (1 - e.hat.sl2)

    cor$pseudo.sl1 <- I(A == a) * S / (p.hat.sl1 * e.hat.sl1) * Y
    cor$inff.sl1 <- I(A == a) * S / (p.hat.sl1 * e.hat.sl1) * (Y - m.hat.sl1) + m.hat.sl1

    cor$pseudo.sl2 <- I(A == a) * S / (p.hat.sl2 * e.hat.sl2) * Y
    cor$inff.sl2 <- I(A == a) * S / (p.hat.sl2 * e.hat.sl2) * (Y - m.hat.sl2) + m.hat.sl2

    ipw1_task <-
      sl3_Task$new(
        cor[fold == 2, ],
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "pseudo.sl1",
        outcome_type = "continuous"
      )

    ipw2_task <-
      sl3_Task$new(
        cor[fold == 1, ],
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "pseudo.sl2",
        outcome_type = "continuous"
      )

    dr1_task <-
      sl3_Task$new(
        cor[fold == 2, ],
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "inff.sl1",
        outcome_type = "continuous"
      )

    dr2_task <-
      sl3_Task$new(
        cor[fold == 1, ],
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "inff.sl2",
        outcome_type = "continuous"
      )

    oracle1_task <-
      sl3_Task$new(
        cor[fold == 2, ],
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "Ya",
        outcome_type = "continuous"
      )

    oracle2_task <-
      sl3_Task$new(
        cor[fold == 1, ],
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "Ya",
        outcome_type = "continuous"
      )

    ipw1_dsl <- delayed_learner_train(sl_cont, ipw1_task)
    ipw2_dsl <- delayed_learner_train(sl_cont, ipw2_task)
    dr1_dsl <- delayed_learner_train(sl_cont, dr1_task)
    dr2_dsl <- delayed_learner_train(sl_cont, dr2_task)
    oracle1_dsl <- delayed_learner_train(sl_cont, oracle1_task)
    oracle2_dsl <- delayed_learner_train(sl_cont, oracle2_task)

    plan(multisession, workers = n.workers)

    ipw1_sched <- Scheduler$new(ipw1_dsl, FutureJob, nworkers = n.workers)
    ipw2_sched <- Scheduler$new(ipw2_dsl, FutureJob, nworkers = n.workers)
    dr1_sched <- Scheduler$new(dr1_dsl, FutureJob, nworkers = n.workers)
    dr2_sched <- Scheduler$new(dr2_dsl, FutureJob, nworkers = n.workers)
    oracle1_sched <- Scheduler$new(oracle1_dsl, FutureJob, nworkers = n.workers)
    oracle2_sched <- Scheduler$new(oracle2_dsl, FutureJob, nworkers = n.workers)

    sl_ipw1 <- ipw1_sched$compute()
    sl_ipw2 <- ipw2_sched$compute()
    sl_dr1 <- dr1_sched$compute()
    sl_dr2 <- dr2_sched$compute()
    sl_oracle1 <- oracle1_sched$compute()
    sl_oracle2 <- oracle2_sched$compute()

    plan(sequential)

    ipw1_task <-
      sl3_Task$new(
        cor,
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "pseudo.sl1",
        outcome_type = "continuous"
      )

    ipw2_task <-
      sl3_Task$new(
        cor,
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "pseudo.sl2",
        outcome_type = "continuous"
      )

    dr1_task <-
      sl3_Task$new(
        cor,
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "inff.sl1",
        outcome_type = "continuous"
      )

    dr2_task <-
      sl3_Task$new(
        cor,
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "inff.sl2",
        outcome_type = "continuous"
      )

    oracle1_task <-
      sl3_Task$new(
        cor,
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "Ya",
        outcome_type = "continuous"
      )

    oracle2_task <-
      sl3_Task$new(
        cor,
        covariates = c("X1", "X2", "X3", "X4"),
        outcome = "Ya",
        outcome_type = "continuous"
      )

    stabwt.sl1 <- sum(mean(I(A == a) * S / (p.hat.sl1 * e.hat.sl1)))

    ipw.sl1 <- sl_ipw1$predict(ipw1_task) / stabwt.sl1
    dr.sl1 <- sl_dr1$predict(dr1_task) / stabwt.sl1 - m.hat.sl1 / stabwt.sl1 + m.hat.sl1
    oracle.sl1 <- sl_oracle1$predict(oracle1_task)

    stabwt.sl2 <- sum(mean(I(A == a) * S / (p.hat.sl2 * e.hat.sl2)))

    ipw.sl2 <- sl_ipw2$predict(ipw2_task) / stabwt.sl2
    dr.sl2 <- sl_dr2$predict(dr2_task)  / stabwt.sl2 - m.hat.sl2 / stabwt.sl2 + m.hat.sl2
    oracle.sl2 <- sl_oracle2$predict(oracle2_task)

  } else {
    m.hat.sl1 <- NA
    m.hat.sl2 <- NA
    ipw.sl1 <- NA
    ipw.sl2 <- NA
    dr.sl1 <- NA
    dr.sl2 <- NA
    oracle.sl1 <- NA
    oracle.sl2 <- NA
  }
  
  mses <-
    sapply(
      list(
        m.hat,
        ipw,
        dr,
        oracle,
        (m.hat.sl1 + m.hat.sl2) / 2,
        (ipw.sl1 + ipw.sl2) / 2,
        (dr.sl1 + dr.sl2) / 2,
        (oracle.sl1 + oracle.sl2) / 2
      ),
      function(x) sqrt(mean((EYa - x)[fold == 3] ^ 2))
    )

  mat <- matrix(
    data = c(mses),
    nrow = 1,
    ncol = 8,
    byrow = TRUE
  )

  return(mat)
}


