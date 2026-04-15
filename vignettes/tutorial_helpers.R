make_binary_fixture <- function(n = 180, seed = 1) {
  set.seed(seed)
  w1 <- rnorm(n)
  w2 <- rnorm(n)
  pi1 <- plogis(0.5 * w1 - 0.3 * w2)
  a_num <- rbinom(n, size = 1, prob = pi1)
  a <- factor(a_num, levels = c(0, 1))
  mu0 <- 0.4 + 0.6 * w1 - 0.25 * w2
  tau <- 0.8 + 0.25 * w1
  mu1 <- mu0 + tau
  y <- rnorm(n, mean = ifelse(a_num == 1, mu1, mu0), sd = 0.6)

  list(
    data = data.frame(Y = y, A = a, W1 = w1, W2 = w2),
    A = a,
    Y = y,
    mu_mat = cbind("0" = mu0, "1" = mu1),
    pi_mat = cbind("0" = 1 - pi1, "1" = pi1)
  )
}

make_multiarm_fixture <- function(n = 220, seed = 2) {
  set.seed(seed)
  w1 <- rnorm(n)
  w2 <- rnorm(n)
  logits <- cbind(
    "0" = 0.1 + 0.2 * w1,
    "1" = -0.1 + 0.35 * w2,
    "2" = 0.15 - 0.2 * w1 + 0.25 * w2
  )
  exp_logits <- exp(logits - apply(logits, 1, max))
  probabilities <- exp_logits / rowSums(exp_logits)
  draws <- apply(probabilities, 1, function(prob) sample.int(3, size = 1, prob = prob))
  a <- factor(c("0", "1", "2")[draws], levels = c("0", "1", "2"))

  mu0 <- 0.2 + 0.4 * w1
  mu1 <- mu0 + 0.6
  mu2 <- mu0 - 0.35 + 0.2 * w2
  mu_mat <- cbind("0" = mu0, "1" = mu1, "2" = mu2)
  y_mean <- mu_mat[cbind(seq_len(n), as.integer(a))]
  y <- rnorm(n, mean = y_mean, sd = 0.5)

  list(
    data = data.frame(Y = y, A = a, W1 = w1, W2 = w2),
    A = a,
    Y = y,
    mu_mat = mu_mat,
    pi_mat = probabilities
  )
}

make_regression_spec <- function() {
  list(
    kind = "regression",
    fit = function(x, y, weights) {
      stats::lm(.outcome ~ ., data = data.frame(.outcome = y, x, check.names = FALSE), weights = weights)
    },
    predict = function(model_fit, newx) {
      as.numeric(stats::predict(model_fit, newdata = as.data.frame(newx)))
    }
  )
}

make_binary_classification_spec <- function() {
  list(
    kind = "classification",
    fit = function(x, y, weights) {
      stats::glm(
        .outcome ~ .,
        data = data.frame(.outcome = as.numeric(as.character(y)), x, check.names = FALSE),
        family = stats::binomial(),
        weights = weights
      )
    },
    predict = function(model_fit, newx) {
      prob1 <- as.numeric(stats::predict(model_fit, newdata = as.data.frame(newx), type = "response"))
      cbind("0" = 1 - prob1, "1" = prob1)
    }
  )
}
