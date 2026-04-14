source(file.path("paper_experiment_scripts", "legacy_config.R"))
legacy_init_paper_env()
legacy_require_packages(c("data.table", "SuperLearner", "sl3", "xgboost", "FKSUM", "drtmle", "origami"))

library(data.table)
library(SuperLearner)
library(sl3)
library(xgboost)
library(FKSUM)
d <- 2
seed_start <- 98103
set.seed(seed_start)

SL.kernelcustom <- function (Y, X, newX, family = gaussian(), obsWeights = rep(1,
                                                                               length(Y)), rangeThresh = 1e-07, ...)
{

  X <- as.matrix(X)
  newX <- as.matrix(newX)
  if(ncol(X) > 1) {
    stop("Univariate X kernel smooths only.")
  }
  fit <- FKSUM::fk_regression(X, Y, type = 'NW' , beta = 1/c(1, 1, 2, 6, 24))
  pred <- predict(fit, xtest = newX)

  fit <- list(object = fit)
  class(fit) <- "SL.kernelcustom"
  out <- list(pred = pred, fit = fit)
  return(out)
}

predict.SL.kernelcustom <- function (object, newdata, ...)
{

  pred <- predict(object, xtest = as.matrix(newdata))

  return(pred)
}


do_sims <- function(n, nsims) {
  nboot <- as.integer(Sys.getenv("CDML_PAPER_NBOOT_KERNEL", unset = "2500"))


  lrnr_kernel <-   Lrnr_pkg_SuperLearner$new("SL.kernelcustom")
  lrnr_kernel <- Lrnr_cv$new(lrnr_kernel)


  sim_results <- lapply(1:nsims, function(i){
    try({
      set.seed(seed_start + i)
      print(paste0("iter: ", i))
      data_list <- get_data(n)
      W <- data_list$W
      A <- data_list$A
      Y <- data_list$Y
      ATE <- data_list$ATE
      n <- length(A)
      nfolds <- 10
      initial_estimators_kernel <- compute_initial_kernel(W,A,Y, folds = nfolds)
      folds <- initial_estimators_kernel$folds
      initial_estimators_misp <- compute_initial(W,A,Y, lrnr_mu =  Lrnr_cv$new(Lrnr_glm$new()), lrnr_pi =  Lrnr_cv$new(Lrnr_glm$new()), folds = folds, stratify_trt = FALSE)

      out_list <- list()

      for(lrnr in c("kernel")) {
        initial_estimators <- initial_estimators_kernel
        for(misp in c("1", "2", "3" )) {
          mu1 <- initial_estimators$mu1
          mu0 <- initial_estimators$mu0
          pi1 <- initial_estimators$pi1
          pi0 <- initial_estimators$pi0
          if(misp == "2") {
            mu1 <- initial_estimators_misp$mu1
            mu0 <- initial_estimators_misp$mu0
          } else if(misp == "3" ) {
            pi1 <- initial_estimators_misp$pi1
            pi0 <- initial_estimators_misp$pi0
          } else if(misp == "4") {
            mu1 <- initial_estimators_misp$mu1
            mu0 <- initial_estimators_misp$mu0
            pi1 <- initial_estimators_misp$pi1
            pi0 <- initial_estimators_misp$pi0
          }

          print(mean(mu1 - mu0))
          out_AIPW <- compute_AIPW(A,Y, mu1=mu1, mu0 =mu0, pi1 = pi1, pi0 = pi0)
          out_AuDRIE <- compute_AuDRIE_boot(A,Y,  mu1=mu1, mu0 =mu0, pi1 = pi1, pi0 = pi0, nboot = nboot, folds = folds, alpha = 0.05)
          out_drtmle <- compute_drtmle(W, A,Y, mu1=mu1, mu0 =mu0, pi1 = pi1, pi0 = pi0, folds = nfolds)


          out <- as.data.table(rbind(
            rbind(unlist(out_AuDRIE), unlist(out_AIPW)),
            unlist(out_drtmle)
          ))
          colnames(out) <- c("estimate", "CI_left", "CI_right")
          out$misp <- misp
          out$estimator <- c("auDRI", "AIPW", "drtmle")
          out$lrnr <- lrnr
          out_list[[paste0(misp, lrnr)]] <- out
        }
      }
      out <- rbindlist(out_list)
      out$n <- n
      out$ATE <- ATE
      out$iter <- i
      return(as.data.table(out))
    })
    return(data.table())
  })
  sim_results <- data.table::rbindlist(sim_results)
  key <- paste0("DR_iter=", nsims, "_n=", n, "_kerneltype=4" )
  try({
    fwrite(sim_results, file.path(legacy_paper_results_dir(), paste0("sim_results_", key, ".csv")))
  })
  return(sim_results)
}


get_data <- function(n, ...) {

  W1 <- runif(n, -2 , 2)
  W2 <- rbinom(n, 1, 0.5)
  W <- cbind(W1, W2)
  colnames(W) <- c("W1", "W2")
  pi0 <- plogis(- W1 + 2 * W1 * W2)
  A <- rbinom(n, 1, pi0)
  mu1 <- plogis(0.2*1 - W1 + 2*W1*W2)
  mu0 <- plogis(0.2*0 - W1 + 2*W1*W2)
  Y <- rbinom(n, 1,ifelse(A==1, mu1, mu0))
  ATE <- 0.03803004

  out <- list(W=W, A = A, Y = Y,  pi0 = pi0, mu1 = mu1, mu0 = mu0, ATE = ATE)

  return(out)
}





compute_AuDRIE_boot <-  function(A,Y, mu1, mu0, pi1, pi0, nboot = 1000, folds, alpha = 0.05) {
  data <- data.table(A, Y, mu1, mu0, pi1, pi0)
  folds <- lapply(folds, `[[`, "validation_set")
  tau_n <- compute_AuDRIE(A,Y, mu1, mu0, pi1, pi0)

  bootstrap_estimates <- sapply(1:nboot, function(iter){
    try({
      bootstrap_indices <- unlist(lapply(folds, function(fold) {
        sample(fold, length(fold), replace = TRUE)
      }))
      data_boot <- data[bootstrap_indices,]
      tau_boot <- compute_AuDRIE(data_boot$A, data_boot$Y, mu1 = data_boot$mu1, mu0 = data_boot$mu0, pi1 = data_boot$pi1, pi0 =data_boot$pi0)
      return(tau_boot)
    })
    return(NULL)
  })




  CI <- tau_n + quantile(bootstrap_estimates - median(bootstrap_estimates), c(alpha/2, 1-alpha/2), na.rm = TRUE)
  return(list(estimate = tau_n, CI = CI))
}



compute_AuDRIE <- function(A,Y, mu1, mu0, pi1, pi0) {
  calibrated_estimators  <- calibrate_nuisances(A,Y, mu1 = mu1, mu0 = mu0, pi1 = pi1, pi0 = pi0)
  mu1_star <- calibrated_estimators$mu1_star
  mu0_star <- calibrated_estimators$mu0_star
  pi1_star <- calibrated_estimators$pi1_star
  pi0_star <- calibrated_estimators$pi0_star

  mu_star <- ifelse(A==1, mu1_star, mu0_star)
  alpha_n <- ifelse(A==1, 1/pi1_star, - 1/pi0_star)
  tau_n <-  mean(mu1_star - mu0_star + alpha_n * (Y - mu_star))
  return(tau_n)
}

compute_drtmle <- function(W, A,Y, mu1, mu0, pi1, pi0, ...) {
  out <- drtmle::drtmle(W = W, A = A, Y = Y, a_0 = c(0,1),
                Qn = list(A0 = mu0, A1 = mu1), gn = list(A0 = pi0, A1= pi1),
                SL_gr = "SL.kernelcustom",
                SL_Qr = "SL.kernelcustom", maxIter = 20)$drtmle
  tau_n <- out$est[2] -  out$est[1]
  se <-  sqrt(as.vector(c(1, -1) %*% out$cov %*% c(1, -1)))
  CI <- tau_n + c(-1,1) * qnorm(1-0.025) * se
  return(list(estimate = tau_n, CI = CI))

}

compute_AIPW <- function(A,Y, mu1, mu0, pi1, pi0) {
  n <- length(A)
  pi1 <- pmax(pi1,  25/(sqrt(n)*log(n)))
  pi0 <- pmax(pi0,  25/(sqrt(n)*log(n)))
  mu <- ifelse(A==1, mu1, mu0)
  alpha_n <- ifelse(A==1, 1/pi1, - 1/pi0)
  tau_n <-  mean(mu1 - mu0 + alpha_n * (Y - mu))
  CI <- tau_n + c(-1,1) * qnorm(1-0.025) * sd(mu1 - mu0 + alpha_n * (Y - mu))/sqrt(n)
  return(list(estimate = tau_n, CI = CI))
}


isoreg_with_xgboost <- function(x,y,max_depth = 15, min_child_weight = 20) {
  data <- xgboost::xgb.DMatrix(data = as.matrix(x), label = as.vector(y))
  iso_fit <- xgboost::xgb.train(params = list(max_depth = max_depth,
                                              min_child_weight = min_child_weight,
                                              monotone_constraints = 1,
                                              eta = 1, gamma = 0,
                                              lambda = 0),
                                data = data, nrounds = 1)
  fun <- function(x) {
    data_pred <- xgboost::xgb.DMatrix(data = as.matrix(x))
    pred <- predict(iso_fit, data_pred)
    return(pred)
  }
  return(fun)
}

calibrate_nuisances <- function(A, Y,mu1, mu0, pi1, pi0) {

  calibrator_mu1 <- isoreg_with_xgboost(mu1[A==1], Y[A==1])
  mu1_star <- calibrator_mu1(mu1)
  calibrator_mu0 <- isoreg_with_xgboost(mu0[A==0], Y[A==0])
  mu0_star <- calibrator_mu0(mu0)


  calibrator_pi1 <- isoreg_with_xgboost(pi1, A)
  pi1_star <- calibrator_pi1(pi1)
  calibrator_pi0 <- isoreg_with_xgboost(pi0, 1-A)
  pi0_star <- calibrator_pi0(pi0)
  return(list(mu1_star= mu1_star, mu0_star=mu0_star, pi1_star = pi1_star, pi0_star = pi0_star))
}


compute_initial_kernel <- function(W,A,Y, folds) {
  library(origami)
  library(FKSUM)
  if(is.numeric(folds)) {
    folds <- origami::folds_vfold(length(A), folds)
  }

  W1 <- W[,1] # continuous
  W2 <- W[,2] # binary

  grid_range <- range(W1)
  cv_fun <- function(fold, ...) {
    train_index <- training(fold = fold)
    val_index <- validation(fold = fold)

    W1_train <- W1[train_index]
    W2_train <- W2[train_index]
    A_train <- A[train_index]
    Y_train <- Y[train_index]
    mu_fit_11 <- FKSUM::fk_regression(W1_train[A_train==1 & W2_train == 1], Y_train[A_train==1 & W2_train == 1],
                                      type = 'NW', h = "cv",
                                      from = grid_range[1], to = grid_range[2])
    mu_fit_10 <- FKSUM::fk_regression(W1_train[A_train==1 & W2_train == 0], Y_train[A_train==1 & W2_train == 0], type = 'NW', h = "cv",
                                      from = grid_range[1], to = grid_range[2])
    mu_fit_01 <- FKSUM::fk_regression(W1_train[A_train==0 & W2_train == 1], Y_train[A_train==0 & W2_train == 1], type = 'NW', h = "cv",
                                      from = grid_range[1], to = grid_range[2])
    mu_fit_00 <- FKSUM::fk_regression(W1_train[A_train==0 & W2_train == 0], Y_train[A_train==0 & W2_train == 0], type = 'NW', h = "cv",
                                      from = grid_range[1], to = grid_range[2])

    pi_fit_1 <- FKSUM::fk_regression(W1_train[W2_train == 1], A_train[W2_train == 1], type = 'NW', h = "cv",
                                     from = grid_range[1], to = grid_range[2])
    pi_fit_0 <- FKSUM::fk_regression(W1_train[W2_train == 0], A_train[W2_train == 0], type = 'NW', h = "cv",
                                     from = grid_range[1], to = grid_range[2])

    W1_val <- W1[val_index]
    W2_val <- W2[val_index]
    A_val <- A[val_index]
    Y_val <- Y[val_index]
    mu11 <- predict(mu_fit_11, xtest = W1_val)
    mu10 <- predict(mu_fit_10, xtest = W1_val)
    mu00 <- predict(mu_fit_00, xtest = W1_val)
    mu01 <- predict(mu_fit_01, xtest = W1_val)
    mu1 <- ifelse(W2_val == 1,mu11, mu10)
    mu0 <- ifelse(W2_val == 1,mu01, mu00)
    pi1_1 <- predict(pi_fit_1,  xtest = W1_val)
    pi1_0 <- predict(pi_fit_0,  xtest = W1_val)
    pi1 <- ifelse(W2_val == 1, pi1_1, pi1_0)
    return(list(index = val_index, mu1 = mu1, mu0 = mu0, pi1 = pi1, pi0 = 1 - pi1))
  }

  out_list <- origami::cross_validate(cv_fun, folds = folds)
  index_order <- order(out_list$index)

  out <- list(mu1 = out_list$mu1[index_order],
              mu0 = out_list$mu0[index_order],
              pi1 = out_list$pi1[index_order],
              pi0 = out_list$pi0[index_order],
              folds = folds)


  print("done")
  return(out)
}

compute_initial <- function(W,A,Y, lrnr_mu, lrnr_pi, folds, stratify_trt = TRUE,   invert = FALSE) {
  data <- data.table(W,A,Y)
  print(lrnr_mu)
  if(stratify_trt) {
    print("stratify")
    taskY0 <- sl3_Task$new(data, covariates = colnames(W), outcome  = "Y", outcome_type = "binomial", folds = folds)
    folds <- taskY0$folds
    fit0 <- lrnr_mu$train(taskY0[A==0])

    mu0 <- as.vector(fit0$predict(taskY0))

    taskY1 <- sl3_Task$new(data, covariates = c(colnames(W)), outcome  = "Y", outcome_type = "binomial", folds = folds)
    fit1 <- lrnr_mu$train(taskY1[A==1])
    mu1 <-  as.vector(fit1$predict(taskY1))

  } else {
    taskY <- sl3_Task$new(data, covariates = c(colnames(W), "A"), outcome  = "Y", outcome_type = "binomial", folds = folds)
    folds <- taskY$folds
    fit <- lrnr_mu$train(taskY)
    data1 <- data0 <- data
    data1$A <- 1
    data0$A <- 0
    taskY1 <- sl3_Task$new(data1, covariates = c(colnames(W), "A"), outcome  = "Y", outcome_type = "binomial", folds = folds)
    taskY0 <- sl3_Task$new(data0, covariates = c(colnames(W), "A"), outcome  = "Y", outcome_type = "binomial", folds = folds)
    mu1 <-  as.vector(fit$predict(taskY1))
    mu0 <-  as.vector(fit$predict(taskY0))
  }


  taskA <- sl3_Task$new(data, covariates = colnames(W), outcome  = "A", folds = folds, outcome_type = "binomial")

  fit1 <- lrnr_pi$train(taskA)
  pi1 <- as.vector(fit1$predict(taskA))
  pi0 <- 1- pi1


  print("done")
  return(list(mu1 = mu1, mu0 = mu0, pi1 = pi1, pi0 = pi0, folds = folds))
}

truncate_pscore_adaptive <- function(A, pi, min_trunc_level = 1e-8) {
  risk_function <- function(cutoff, level) {
    pi <- pmax(pi, cutoff)
    pi <- pmin(pi, 1 - cutoff)
    alpha <- A/pi - (1-A)/(1-pi) #Riesz-representor
    alpha1 <- 1/pi
    alpha0 <- - 1/(1-pi)
    mean(alpha^2 - 2*(alpha1 - alpha0))
  }
  cutoff <- optim(1e-5, fn = risk_function, method = "Brent", lower = min_trunc_level, upper = 0.5, level = 1)$par
  pi <- pmin(pi, 1 - cutoff)
  pi <- pmax(pi, cutoff)
  pi
}

