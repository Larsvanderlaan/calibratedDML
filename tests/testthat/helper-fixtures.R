oracle_binary_fixture <- function(n = 400, seed = 1) {
  set.seed(seed)
  w1 <- rnorm(n)
  w2 <- rnorm(n)
  pi1 <- plogis(0.6 * w1 - 0.4 * w2)
  a <- rbinom(n, size = 1, prob = pi1)
  mu0 <- 0.5 + w1 - 0.5 * w2
  tau <- 1 + 0.4 * w1
  mu1 <- mu0 + tau
  y <- rnorm(n, mean = ifelse(a == 1, mu1, mu0), sd = 1)

  list(
    data = data.frame(Y = y, A = a, W1 = w1, W2 = w2),
    A = a,
    Y = y,
    mu_mat = cbind("0" = mu0, "1" = mu1),
    pi_mat = cbind("0" = 1 - pi1, "1" = pi1),
    ey_truth = c("EY0" = mean(mu0), "EY1" = mean(mu1)),
    contrast_truth = c("ATE1" = mean(tau)),
    ate = mean(tau)
  )
}

oracle_multiarm_fixture <- function(n = 450, seed = 2) {
  set.seed(seed)
  w1 <- rnorm(n)
  w2 <- rnorm(n)
  logits <- cbind(
    control = 0,
    treat1 = 0.5 * w1,
    treat2 = -0.25 * w1 + 0.5 * w2
  )
  exp_logits <- exp(logits - apply(logits, 1, max))
  probabilities <- exp_logits / rowSums(exp_logits)
  draws <- apply(probabilities, 1, function(prob) sample.int(3, size = 1, prob = prob))
  levels <- c("0", "1", "2")
  a <- factor(levels[draws], levels = levels)
  mu0 <- 0.2 + 0.5 * w1 - 0.25 * w2
  mu1 <- mu0 + 0.8 + 0.2 * w1
  mu2 <- mu0 - 0.4 + 0.3 * w2
  mu_mat <- cbind("0" = mu0, "1" = mu1, "2" = mu2)
  y_mean <- mu_mat[cbind(seq_len(n), as.integer(a))]
  y <- rnorm(n, mean = y_mean, sd = 1)

  colnames(probabilities) <- levels

  list(
    data = data.frame(Y = y, A = a, W1 = w1, W2 = w2),
    A = a,
    Y = y,
    mu_mat = mu_mat,
    pi_mat = probabilities,
    ey_truth = c("EY0" = mean(mu0), "EY1" = mean(mu1), "EY2" = mean(mu2)),
    contrast_truth = c("ATE1" = mean(mu1 - mu0), "ATE2" = mean(mu2 - mu0))
  )
}

oracle_binary_nonlinear_fixture <- function(n = 400, seed = 3) {
  set.seed(seed)
  w1 <- rnorm(n)
  w2 <- rnorm(n)
  pi1 <- plogis(0.4 * sin(w1) + 0.35 * w2)
  a <- rbinom(n, size = 1, prob = pi1)
  mu0 <- 0.2 + 0.8 * sin(w1) + 0.3 * w2^2
  tau <- 0.7 + 0.25 * w1 - 0.15 * w2
  mu1 <- mu0 + tau
  y <- rnorm(n, mean = ifelse(a == 1, mu1, mu0), sd = 1)

  list(
    data = data.frame(Y = y, A = a, W1 = w1, W2 = w2),
    A = a,
    Y = y,
    mu_mat = cbind("0" = mu0, "1" = mu1),
    pi_mat = cbind("0" = 1 - pi1, "1" = pi1),
    ey_truth = c("EY0" = mean(mu0), "EY1" = mean(mu1)),
    contrast_truth = c("ATE1" = mean(tau)),
    ate = mean(tau)
  )
}

weighted_binary_fixture <- function(n = 400, seed = 11) {
  fixture <- oracle_binary_fixture(n = n, seed = seed)
  weights <- exp(0.3 * fixture$data$W1 - 0.15 * fixture$data$W2)
  weights <- weights / mean(weights)
  mu0 <- fixture$mu_mat[, "0"]
  mu1 <- fixture$mu_mat[, "1"]
  fixture$data$weight <- weights
  fixture$sample_weight <- weights
  fixture$ey_truth <- c("EY0" = weighted.mean(mu0, weights), "EY1" = weighted.mean(mu1, weights))
  fixture$contrast_truth <- c("ATE1" = weighted.mean(mu1 - mu0, weights))
  fixture$ate <- unname(fixture$contrast_truth[[1]])
  fixture
}

weighted_multiarm_fixture <- function(n = 450, seed = 13) {
  fixture <- oracle_multiarm_fixture(n = n, seed = seed)
  weights <- exp(0.25 * fixture$data$W1 + 0.1 * fixture$data$W2)
  weights <- weights / mean(weights)
  fixture$data$weight <- weights
  fixture$sample_weight <- weights
  fixture$ey_truth <- c(
    "EY0" = weighted.mean(fixture$mu_mat[, "0"], weights),
    "EY1" = weighted.mean(fixture$mu_mat[, "1"], weights),
    "EY2" = weighted.mean(fixture$mu_mat[, "2"], weights)
  )
  fixture$contrast_truth <- c(
    "ATE1" = weighted.mean(fixture$mu_mat[, "1"] - fixture$mu_mat[, "0"], weights),
    "ATE2" = weighted.mean(fixture$mu_mat[, "2"] - fixture$mu_mat[, "0"], weights)
  )
  fixture
}
