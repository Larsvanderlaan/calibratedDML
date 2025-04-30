#' Calibrated Debiased Machine Learning (CDML) for Average Treatment Effect Estimation
#'
#' This function estimates the average treatment effect (ATE) using Calibrated Debiased Machine Learning (CDML).
#' The CDML approach leverages flexible machine learning algorithms for estimating the propensity score
#' and outcome models, resulting in a doubly robust and asymptotically linear estimator.
#' The method integrates calibration, cross-fitting, and debiased estimation techniques to
#' provide consistent estimation and inference, such as confidence intervals and hypothesis testing,
#' even in the presence of potential misspecification or slow convergence of nuisance function estimators.
#'
#' @param W A matrix or data frame of covariates.
#' @param A A vector representing the treatment variable (binary indicator).
#' @param Y A vector representing the outcome variable.
#' @param weights Optional weights for each observation. Defaults to equal weights if not provided.
#' @param learners_treatment Optional list of `sl3_Learner` objects for propensity score estimation.
#'   If not provided, defaults to `default_learners`. These `sl3_Learner` objects are from the `tlverse/sl3` package.
#' @param learners_outcome Optional list of `sl3_Learner` objects for outcome estimation.
#'   If not provided, defaults to `default_learners`. These `sl3_Learner` objects are from the `tlverse/sl3` package.
#' @param pi_mat Optional matrix of propensity scores, where the first column represents \eqn{P(A=1|W)}
#'   and the second column represents \eqn{P(A=0|W)}. If not provided, it will be estimated using
#'   the specified learners for the propensity score.
#' @param mu_mat Optional matrix of outcome predictions, where the first column represents the
#'   predicted outcome for \eqn{A=1} and the second column represents the predicted outcome for \eqn{A=0}.
#'   If not provided, it will be estimated using the specified learners for the outcome.
#' @param nboot Number of bootstrap samples used to estimate confidence intervals. Defaults to 1000.
#' @param alpha Significance level for confidence intervals. Defaults to 0.05 (corresponding to 95\% confidence intervals).
#'
#' @return A list containing the estimated average treatment effect (ATE) and corresponding confidence intervals.
#' @examples
#' \dontrun{
#' # Example usage with sample data
#' W <- matrix(rnorm(1000), ncol = 5)
#' A <- rbinom(200, 1, 0.5)
#' Y <- rnorm(200)
#' results <- cdml(W, A, Y)
#' print(results)
#' }
#' @export
cdml <- function(W, A, Y, weights = NULL, learners_treatment = NULL, learners_outcome = NULL, pi_mat = NULL, mu_mat = NULL, nboot = 1000, alpha = 0.05){

  if(is.null(learners_treatment)) {
    learners_treatment <- default_learners()
  }
  if(is.null(learners_outcome)) {
    learners_outcome <- default_learners()
  }
  learners_treatment <- make_learner(Pipeline, Lrnr_cv$new(Stack$new(learners_treatment)), Lrnr_cv_selector$new(loss_loglik_binomial))
  learners_outcome <- make_learner(Pipeline, Lrnr_cv$new(Stack$new(learners_outcome)), Lrnr_cv_selector$new(loss_squared_error))

  if(is.null(weights)) {
    weights <- rep(1, length(A))
  }
  W <- as.matrix(W)
  if(is.null(colnames(W))){
    colnames(W) <- paste0("W", 1:ncol(W))
  }
  A <- as.vector(A)
  Y <- as.vector(Y)
  data = data.table(W, A = A, Y = Y, weights = weights)

  if(is.null(pi_mat)) {
    task_A <- sl3_Task$new(data, covariates = colnames(W), outcome = "A", weights = "weights")
    learners_treatment <- learners_treatment$train(task_A)
    pi1 <- learners_treatment$predict(task_A)
    pi_mat <- cbind(pi1, pi0)
  }
  if(is.null(mu_mat)) {
    data1 <- data.table::copy(data)
    data1$A <- 1
    data0 <- data.table::copy(data)
    data0$A <- 0
    task_Y <- sl3_Task$new(data, covariates = c(colnames(W), "A"), outcome = "Y", weights = "weights")
    task_Y1 <- sl3_Task$new(data1, covariates = c(colnames(W), "A"), outcome = "Y", weights = "weights")
    task_Y0 <- sl3_Task$new(data0, covariates = c(colnames(W), "A"), outcome = "Y", weights = "weights")
    learners_outcome <- learners_outcome$train(task_Y)
    mu1 <- learners_outcome$predict(task_Y1)
    mu0 <- learners_outcome$predict(task_Y0)
    mu_mat <- cbind(mu1, mu0)
  }

  pi1 <- pi_mat[,1]
  pi0 <- pi_mat[,2]
  mu1 <- mu_mat[,1]
  mu0 <- mu_mat[,2]

  output <- bootstrap_cdml_ate(A, Y, mu1, mu0, pi1, pi0, weights, nboot = nboot, folds = NULL, alpha = alpha)
  return(output)
}


#' Compute Calibrated Debiased Machine Learning Estimate (ICDML)
#'
#' Calculates the Calibrated Debiased Machine Learning (C-DML) estimator for
#' causal inference, specifically for estimating linear functionals of the outcome regression.
#'
#' @param A A binary indicator variable (1 for treatment, 0 for control).
#' @param Y Observed outcomes.
#' @param mu1 Cross-fitted outcome regression estimates for the treated group.
#' @param mu0 Cross-fitted outcome regression estimates for the control group.
#' @param pi1 Cross-fitted propensity score estimates for the treatment group (\code{A = 1}).
#' @param pi0 Cross-fitted propensity score estimates for the control group (\code{A = 0}).
#' @param weights Optional weights to use in the estimation. Defaults to equal weights if \code{NULL}.
#' @param functional Function to compute the desired functional of interest, given relevant inputs.
#' @param representer Function to compute the debiasing term for the estimator.
#'
#' @return A numeric value representing the estimated causal effect using the ICDML estimator.
#'
#' @details
#' The function computes a C-DML estimate, which involves estimating a "plug-in" term and a
#' debiasing term based on cross-fitted outcome regressions and propensity scores. This function
#' incorporates doubly robust properties to mitigate estimation bias and improve robustness.
#'
#' @seealso \code{\link{estimate_cdml}}, \code{\link{calibrate_outcome_regression}}, \code{\link{calibrate_inverse_weights}}
#'
#' @export
estimate_dml <- function(A, Y, mu1, mu0, pi1, pi0, weights = NULL, functional = NULL, representer = NULL) {
  if (is.null(weights)) {
    weights <- rep(1, length(A)) # Assign equal weights if none are provided
  }

  mu <- ifelse(A == 1, mu1, mu0) # Determine the relevant outcome estimate based on treatment status
  estimate_plugin <- functional(mu1 = mu1, mu0 = mu0, A = A, Y = Y, weights = weights) # Compute the functional of interest

  # Compute the debiasing representer term
  alpha_n <- representer(pi1 = pi1, pi0 = pi0, A = A, weights = weights)

  # Calculate the ICDML estimate using weighted mean
  estimate <- weighted.mean(estimate_plugin + alpha_n * (Y - mu), weights = weights)

  return(estimate) # Return the estimated causal effect
}

#' Estimate Calibrated Debiased Machine Learning (CDML)
#'
#' Extends the DML estimate by calibrating outcome regression estimates and propensity scores.
#' Implements the calibrated version of the CDML estimator for causal inference.
#'
#' @param A A binary indicator variable (1 for treatment, 0 for control).
#' @param Y Observed outcomes.
#' @param mu1 Cross-fitted outcome regression estimates for the treated group.
#' @param mu0 Cross-fitted outcome regression estimates for the control group.
#' @param pi1 Cross-fitted propensity score estimates for the treatment group (\code{A = 1}).
#' @param pi0 Cross-fitted propensity score estimates for the control group (\code{A = 0}).
#' @param weights Optional weights to use in the estimation. Defaults to equal weights if \code{NULL}.
#' @param functional Function to compute the desired functional of interest, given relevant inputs.
#' @param representer Function to compute the debiasing term for the estimator.
#'
#' @return The estimated causal effect using the CDML estimator.
#'
#' @details
#' This function calibrates the outcome regression and inverse propensity weights before
#' using the CDML framework for estimation. Calibration improves estimation accuracy and
#' robustness, leveraging cross-fitting and doubly robust methods.
#'
#' @seealso \code{\link{estimate_dml}}, \code{\link{calibrate_outcome_regression}}, \code{\link{calibrate_inverse_weights}}
#'
#' @export
estimate_cdml <- function(A, Y, mu1, mu0, pi1, pi0, weights = NULL, functional = NULL, representer = NULL) {
  if (is.null(weights)) {
    weights <- rep(1, length(A)) # Assign equal weights if none are provided
  }

  # Calibrate outcome regression estimates
  calibrated_outcome <- calibrate_outcome_regression(Y, mu1, mu0, A, weights = weights)
  mu1_star <- calibrated_outcome$mu1_star
  mu0_star <- calibrated_outcome$mu0_star
  mu_star <- ifelse(A == 1, mu1_star, mu0_star) # Calibrated outcome estimates

  # Calibrate inverse propensity weights
  calibrated_weights <- calibrate_inverse_weights(A, pi1, pi0, weights = weights)
  pi1_star <- calibrated_weights['pi1_star']
  pi0_star <- calibrated_weights['pi0_star']

  # Estimate causal effect using calibrated values
  estimate <- estimate_dml(A, Y, mu1, mu0, pi1, pi0, weights = weights, functional = functional, representer = representer)
  return(estimate)
}

# Additional functions with similar documentation and comments provided for completeness.




#' Compute Calibrated Debiased Machine Learning Estimate (ICDML) for Average Treatment Effect
#'
#' Calculates the Calibrated Debiased Machine Learning (C-DML) estimator for
#' causal inference, specifically estimating the Average Treatment Effect (ATE).
#'
#' @param A A binary indicator variable (1 for treatment, 0 for control).
#' @param Y Observed outcomes.
#' @param mu1 Cross-fitted outcome regression estimates for the treated group.
#' @param mu0 Cross-fitted outcome regression estimates for the control group.
#' @param pi1 Cross-fitted propensity score estimates for the treatment group (\code{A = 1}).
#' @param pi0 Cross-fitted propensity score estimates for the control group (\code{A = 0}).
#' @param weights Optional weights to use in the estimation. Defaults to equal weights if \code{NULL}.
#'
#' @return A numeric value representing the estimated causal effect using the ICDML estimator.
#'
#' @details
#' This function calibrates the outcome regression estimates and the inverse propensity weights
#' before computing the CDML estimator. It uses cross-fitting and calibration techniques to achieve
#' doubly robust estimation of causal effects.
#'
#' @seealso \code{\link{estimate_dml}}, \code{\link{calibrate_outcome_regression}}, \code{\link{calibrate_inverse_weights}}
#'
#' @export
estimate_cdml_ate <- function(A, Y, mu1, mu0, pi1, pi0, weights = NULL) {
  if (is.null(weights)) {
    weights <- rep(1, length(A)) # Assign equal weights if none are provided
  }

  # Calibrate outcome regression estimates
  calibrated_outcome <- calibrate_outcome_regression(Y, mu1, mu0, A, weights = weights)
  mu1_star <- calibrated_outcome$mu1_star
  mu0_star <- calibrated_outcome$mu0_star
  mu_star <- ifelse(A == 1, mu1_star, mu0_star) # Calibrated outcome regression based on treatment

  # Calibrate inverse propensity weights
  calibrated_weights <- calibrate_inverse_weights(A, pi1, pi0, weights = weights)
  pi1_star <- calibrated_weights['pi1_star']
  pi0_star <- calibrated_weights['pi0_star']

  # Estimate using CDML with various functionals
  estimates <- sapply(functionals_info, function(info) {
    estimate_dml(A, Y, mu1, mu0, pi1, pi0, weights = weights, functional = info[['fun']], representer = info[['rep']])
  })
  names(estimates) <- names(functionals_info) # Name the estimates based on the functionals
  return(estimates)
}

#' Bootstrap Calibrated Debiased Machine Learning Estimate (ICDML)
#'
#' Computes a bootstrap-assisted version of the ICDML estimator to obtain a confidence interval
#' for the causal effect, leveraging calibrated debiased machine learning.
#'
#' @param A A binary indicator variable (1 for treatment, 0 for control).
#' @param Y Observed outcomes.
#' @param mu1 Predicted outcome for the treated group.
#' @param mu0 Predicted outcome for the control group.
#' @param pi1 Estimated propensity score for the treatment group (\code{A = 1}).
#' @param pi0 Estimated propensity score for the control group (\code{A = 0}).
#' @param nboot Integer. Number of bootstrap samples (default is 1000).
#' @param folds List or \code{NULL}. A list of cross-validation folds, where each fold contains a validation set of indices. If \code{NULL}, a simple random sampling is used for each bootstrap iteration.
#' @param alpha Significance level for the confidence interval (default is 0.05).
#'
#' @return A list with the following elements:
#' \describe{
#'   \item{tau_n}{Estimated causal effect using ICDML.}
#'   \item{CI}{Bootstrap confidence interval for \code{tau_n}.}
#' }
#'
#' @details
#' This function estimates the causal effect using the ICDML estimator and computes a bootstrap confidence
#' interval by resampling within cross-validation folds. If \code{folds} is \code{NULL}, a simple random sample is drawn for each bootstrap iteration. Cross-fitting with bootstrapping allows for doubly robust inference, even when some nuisance function estimates are inconsistent.
#'
#' @seealso \code{\link{estimate_cdml_ate}}, \code{\link{calibrate_outcome_regression}}, \code{\link{calibrate_inverse_weights}}
#'
#' @export
bootstrap_cdml_ate <- function(A, Y, mu1, mu0, pi1, pi0, weights, nboot = 1000, folds = NULL, alpha = 0.05) {
  # Load data
  data <- data.table::data.table(A, Y, mu1, mu0, pi1, pi0, weights = weights)

  # Calculate the ICDML point estimate
  estimates <- estimate_cdml_ate(A, Y, mu1, mu0, pi1, pi0, weights = weights)

  # Perform bootstrap resampling
  bootstrap_estimates <- do.call(rbind, lapply(1:nboot, function(iter) {
    tryCatch({
      # Determine bootstrap indices
      if (is.null(folds)) {
        bootstrap_indices <- sample(seq_len(nrow(data)), nrow(data), replace = TRUE) # Random sample when no folds
      } else {
        bootstrap_indices <- unlist(lapply(folds, function(fold) {
          sample(fold, length(fold), replace = TRUE) # Sample within each fold if provided
        }))
      }

      # Subset data for bootstrap
      data_boot <- data[bootstrap_indices,]
      estimates_boot <- estimate_cdml_ate(data_boot$A, data_boot$Y,
                                          mu1 = data_boot$mu1, mu0 = data_boot$mu0,
                                          pi1 = data_boot$pi1, pi0 = data_boot$pi0,
                                          weights = data_boot$weights)
      return(estimates_boot)
    }, error = function(e) NA) # Return NA on error
  }))

  # Remove NA values from bootstrap estimates
  bootstrap_estimates <- na.omit(bootstrap_estimates)
  standard_error <- apply(bootstrap_estimates, 2, sd)

  # Calculate the confidence interval
  CI <- do.call(rbind, lapply(1:length(estimates), function(index) {
    estimates[index] + quantile(bootstrap_estimates[, index] - estimates[index], probs = c(alpha / 2, 1 - alpha / 2), na.rm = TRUE)
  }))

  # Output results
  output <- data.frame(estimand = names(estimates), estimate = estimates, standard_error = standard_error, lower_bound = CI[, 1], upper_bound = CI[, 2], significance = 1 - alpha)
  return(output)
}

#' Bootstrap Calibrated Debiased Machine Learning Estimate (CDML)
#'
#' Computes a general bootstrap-assisted version of the CDML estimator, supporting custom functional forms and representers.
#'
#' @param A A binary indicator variable (1 for treatment, 0 for control).
#' @param Y Observed outcomes.
#' @param mu1 Cross-fitted outcome regression estimates for the treated group.
#' @param mu0 Cross-fitted outcome regression estimates for the control group.
#' @param pi1 Cross-fitted propensity score estimates for the treatment group (\code{A = 1}).
#' @param pi0 Cross-fitted propensity score estimates for the control group (\code{A = 0}).
#' @param weights Optional weights to use in the estimation.
#' @param nboot Integer. Number of bootstrap samples (default is 1000).
#' @param folds List or \code{NULL}. A list of cross-validation folds for resampling.
#' @param alpha Significance level for the confidence interval (default is 0.05).
#' @param functional A function defining the target functional for estimation.
#' @param representer A function defining the representer for debiasing.
#'
#' @return A data frame with point estimates, standard errors, and confidence intervals for each functional.
#'
#' @details
#' This function generalizes the CDML bootstrap by supporting custom functionals and representers,
#' making it applicable to a broader range of causal estimands.
#'
#' @seealso \code{\link{estimate_cdml}}, \code{\link{calibrate_outcome_regression}}, \code{\link{calibrate_inverse_weights}}
#'
#' @export
bootstrap_cdml <- function(A, Y, mu1, mu0, pi1, pi0, weights, nboot = 1000, folds = NULL, alpha = 0.05, functional = functional, representer = representer) {
  # Load data
  data <- data.table::data.table(A, Y, mu1, mu0, pi1, pi0, weights = weights)

  # Calculate the initial point estimate
  estimate <- estimate_cdml_ate(A, Y, mu1, mu0, pi1, pi0, weights = weights)

  # Perform bootstrap resampling
  bootstrap_estimates <- sapply(1:nboot, function(iter) {
    tryCatch({
      # Determine bootstrap indices
      if (is.null(folds)) {
        bootstrap_indices <- sample(seq_len(nrow(data)), nrow(data), replace = TRUE) # Random sample when no folds
      } else {
        bootstrap_indices <- unlist(lapply(folds, function(fold) {
          sample(fold, length(fold), replace = TRUE) # Sample within each fold if provided
        }))
      }

      # Subset data for bootstrap
      data_boot <- data[bootstrap_indices,]
      estimates_boot <- estimate_cdml(data_boot$A, data_boot$Y,
                                      mu1 = data_boot$mu1, mu0 = data_boot$mu0,
                                      pi1 = data_boot$pi1, pi0 = data_boot$pi0,
                                      weights = data_boot$weights,
                                      functional = functional, representer = representer)
      return(estimates_boot)
    }, error = function(e) NA) # Return NA on error
  })

  # Remove NA values from bootstrap estimates
  bootstrap_estimates <- na.omit(bootstrap_estimates)
  standard_error <- sd(bootstrap_estimates)

  # Calculate the confidence interval
  CI <- estimate + quantile(bootstrap_estimates - estimate, probs = c(alpha / 2, 1 - alpha / 2), na.rm = TRUE)

  # Output results
  output <- data.frame(estimand = names(estimate), estimate = estimate, standard_error = standard_error, lower_bound = CI[1], upper_bound = CI[2], significance = 1 - alpha)
  return(output)
}
