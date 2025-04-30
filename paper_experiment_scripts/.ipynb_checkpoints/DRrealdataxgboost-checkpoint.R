library(data.table)
library(sl3)
library(xgboost)
set.seed(98103)



# = c("lalonde_cps", "lalonde_psid", "twins")
do_real_data <- function(data_name) {
  print(data_name)



  stack_all <- Stack$new(
    list(
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 1, nrounds = 20, eta = 0.3 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 2, nrounds = 20, eta = 0.3 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 3, nrounds = 20, eta = 0.3 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 4, nrounds = 20, eta = 0.3 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 5, nrounds = 20, eta = 0.3 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 6, nrounds = 20, eta = 0.3 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 1, nrounds = 20, eta = 0.25 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 2, nrounds = 20, eta = 0.25 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 3, nrounds = 20, eta = 0.25 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 4, nrounds = 20, eta = 0.25 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 5, nrounds = 20, eta = 0.25 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 6, nrounds = 20, eta = 0.25 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 1, nrounds = 20, eta = 0.2 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 2, nrounds = 20, eta = 0.2 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 3, nrounds = 20, eta = 0.2 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 4, nrounds = 20, eta = 0.2 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 5, nrounds = 20, eta = 0.2 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 6, nrounds = 20, eta = 0.2 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 1, nrounds = 20, eta = 0.15 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 2, nrounds = 20, eta = 0.15 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 3, nrounds = 20, eta = 0.15 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 4, nrounds = 20, eta = 0.15 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 5, nrounds = 20, eta = 0.15 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 6, nrounds = 20, eta = 0.15 )
    )
  )

  # stack_all <- Stack$new(
  #   list(
  #     Lrnr_cvxgboost$new(min_child_weight = 5, max_depth = 1, nrounds = 100, eta = 0.2 ),
  #     Lrnr_cvxgboost$new(min_child_weight = 5, max_depth = 2, nrounds = 100, eta = 0.15 ),
  #     Lrnr_cvxgboost$new(min_child_weight = 5, max_depth = 3, nrounds = 100, eta = 0.15 ),
  #     Lrnr_cvxgboost$new(min_child_weight = 5, max_depth = 4, nrounds = 100, eta = 0.1 ),
  #     Lrnr_cvxgboost$new(min_child_weight = 5, max_depth = 5, nrounds = 100, eta = 0.1 ),
  #     Lrnr_cvxgboost$new(min_child_weight = 5, max_depth = 6, nrounds = 100, eta = 0.1 )
  #
  #   )
  # )
  if(length(grep("acic2017", data_name)) > 0 ) {
    stack_all <- make_learner(Pipeline, Lrnr_screener_coefs$new( Lrnr_glmnet$new()) , stack_all)
  } else {
    stack_all <- make_learner(Pipeline, Lrnr_screener_coefs$new( Lrnr_glmnet$new()) , stack_all)
  }



  lrnr_mu_all <-  Pipeline$new(Lrnr_cv$new(stack_all), Lrnr_cv_selector$new(loss_squared_error))
  lrnr_pi_all <- Pipeline$new(Lrnr_cv$new(stack_all), Lrnr_cv_selector$new(loss_squared_error))


  iters <- 0:99
  if(length(grep("acic2017", data_name)) > 0 ){
    iters <- 1:250
  }
  if(data_name %in% c("ihdp") ){
    iters <- 1:100
  }
  # acic2018_1000 acic2018_2500 acic2018_5000 acic2018_10000
  if(length(grep("acic2018", data_name)) > 0 ) {
    nsize <- as.numeric(gsub("acic2018_", "", data_name))
    params <- fread("~/DRinference/data/scaling_small/params.csv")
    ids <- params$ufid[params$size == nsize]
    iters <- seq_along(ids)
  }

  print(iters)

  sim_results <- lapply(iters, function(i){
    try({
      print(paste0("iter: ", i))
      if(data_name == "ihdp") {
        data <- fread(paste0("~/DRinference/data/ihdp/ihdp_npci_", i, ".csv"))

        #data <- fread(paste0("~/DRinference/data/ihdp/ihdp_npci_", i, ".csv"))
        covariates <- setdiff(names(data), c( "t", "y", "ate"))
        W <- as.matrix(data[, covariates, with = FALSE])
        A <- data[, "t", with = FALSE][[1]]
        Y <- data[, "y", with = FALSE][[1]]
        ATE <- mean(data$ate)
      } else if(length(grep("acic2018", data_name)) > 0 ) {
        id <- ids[i]
        f <- fread(paste0("~/DRinference/data/scaling_small/factuals/", id, ".csv"))
        x <- fread(paste0("~/DRinference/data/scaling_small/x.csv"))
        x <- x[match(f$sample_id, x$sample_id)]
        x$sample_id <- NULL
        ATE <- params$effect_size[params$ufid == id]
        W <- as.matrix(x)
        A <- f$z
        Y <- f$y
      } else if(length(grep("acic2017", data_name)) > 0 ) {
        sim_id <- as.numeric(gsub("acic2017_", "", data_name))
        print(sim_id)
        if(!require(aciccomp2017)) {
          devtools::install_github("vdorie/aciccomp/2017")
        }
        library(aciccomp2017)
        W <- aciccomp2017::input_2017
        data <- dgp_2017(sim_id, i)
        A <- data$z
        Y <- data$y
        ATE <- mean(data$alpha)
      }
      else {
        link <- "https://raw.githubusercontent.com/bradyneal/realcause/master/realcause_datasets/"
        data <- fread(paste0(link, data_name, "_sample", i, ".csv"))

        covariates <- setdiff(names(data), c( "t", "y", "y0", "y1", "ite"))
        W <- as.matrix(data[, covariates, with = FALSE])
        A <- data[, "t", with = FALSE][[1]]
        Y <- data[, "y", with = FALSE][[1]]
        ATE <- mean(data$ite)
      }
      print(dim(W))

      n <- length(A)
      folds <- 10
      initial_estimators_all <- compute_initial(W,A,Y, lrnr_mu = lrnr_mu_all, lrnr_pi = lrnr_pi_all, folds = folds, invert = FALSE)
      folds <- initial_estimators_all$folds
      initial_estimators_misp <- compute_initial(W,A,Y, lrnr_mu =  Lrnr_cv$new(Lrnr_mean$new()), lrnr_pi =  Lrnr_cv$new(Lrnr_mean$new()), folds = folds)

      out_list <- list()

      for(lrnr in c( "all")) {
        print(lrnr)
        if(lrnr == "all") {
          initial_estimators <- initial_estimators_all
        }
        for(misp in c("1", "2", "3")) {
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


          out_AIPW <- compute_AIPW(A,Y, mu1=mu1, mu0 =mu0, pi1 = pi1, pi0 = pi0)

          out_AuDRIE <- compute_AuDRIE_boot(A,Y,  mu1=mu1, mu0 =mu0, pi1 = pi1, pi0 = pi0, nboot = 1000, folds = folds, alpha = 0.05)

          out <- as.data.table(rbind(unlist(out_AuDRIE), unlist(out_AIPW)))
          colnames(out) <- c("estimate", "CI_left", "CI_right")
          out$misp <- misp
          out$estimator <- c("auDRI", "AIPW")
          out$lrnr <- lrnr
          out_list[[paste0(misp, lrnr)]] <- out
        }
      }
      print(out$estimate - ATE)
      out <- rbindlist(out_list)
      out$n <- n
      out$ATE <- ATE
      out$iter <- i
      return(as.data.table(out))
    })
    return(data.table())
  })
  sim_results <- data.table::rbindlist(sim_results)
  key <- data_name
  try({fwrite(sim_results, paste0("~/DRinference/simResultsDR/sim_results_", key, "_xgboost.csv"))})
  return(sim_results)
}



compute_AuDRIE_boot <-  function(A,Y, mu1, mu0, pi1, pi0, nboot = 5000, folds, alpha = 0.05) {
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

isoreg_with_xgboost <- function(x,y,max_depth = 15, min_child_weight = 10) {
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

compute_AIPW <- function(A,Y, mu1, mu0, pi1, pi0) {
  n <- length(A)

  pi1 <- pmax(pi1,  25/(sqrt(n)*log(n)))
  pi0 <- pmax(pi0,  25/(sqrt(n)*log(n)))

  #pi1 <- truncate_pscore_adaptive(A, pi1)
  #pi0 <- truncate_pscore_adaptive(1-A, pi0)

  mu <- ifelse(A==1, mu1, mu0)
  alpha_n <- ifelse(A==1, 1/pi1, - 1/pi0)
  tau_n <-  mean(mu1 - mu0 + alpha_n * (Y - mu))
  se <- sd(mu1 - mu0 + alpha_n * (Y - mu))/sqrt(n)

  CI <- tau_n + c(-1,1) * qnorm(1-0.025) * se
  return(list(estimate = tau_n, CI = CI))
}




calibrate_nuisances <- function(A, Y,mu1, mu0, pi1, pi0) {
  # mu <- ifelse(A==1, mu1 , mu0)
  # calibrator_mu <- isoreg_with_xgboost(mu, Y)
  # mu1_star <- calibrator_mu(mu1)
  # mu0_star <- calibrator_mu(mu0)
  # print(quantile(mu1))
  # print(quantile(mu1_star))
  # print(table(mu1_star))
  # print(table(mu0_star))


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





compute_initial <- function(W,A,Y, lrnr_mu, lrnr_pi, folds,   invert = FALSE) {
  data <- data.table(W,A,Y)

  taskY0 <- sl3_Task$new(data, covariates = colnames(W), outcome  = "Y"  ,folds = folds)
  folds <- taskY0$folds
  fit0 <- lrnr_mu$train(taskY0[A==0])
  mu0 <- fit0$predict(taskY0)
  data$offset <- mu0
  taskY1 <- sl3_Task$new(data, covariates = c(colnames(W), "offset"), outcome  = "Y"  ,folds = folds)
  fit1 <- lrnr_mu$train(taskY1[A==1])
  mu1 <-  fit1$predict(taskY1)

  print(fit0$fit_object$learner_fits$Lrnr_cv_selector_NULL$fit_object$cv_risk)

  print(fit1$fit_object$learner_fits$Lrnr_cv_selector_NULL$fit_object$cv_risk)




  taskA <- sl3_Task$new(data, covariates = colnames(W), outcome  = "A", folds = folds, outcome_type = "binomial")

  fit1 <- lrnr_pi$train(taskA)
  pi1 <- fit1$predict(taskA)
  print(fit1$fit_object$learner_fits$Lrnr_cv_selector_NULL$fit_object$cv_risk)
  if(invert) {

    data0 <- data
    data0$A <- 1-A
    taskA0 <- sl3_Task$new(data0, covariates = colnames(W), outcome  = "A", folds = folds, outcome_type = "continuous")
    fit0 <- lrnr_pi$train(taskA0)
    pi0 <- fit0$predict(taskA0)
    print(fit0$fit_object$learner_fits$Lrnr_cv_selector_NULL$fit_object$cv_risk)


    pi1 <- 1/pi1
    pi0 <- 1/pi0
  } else {
    pi0 <- 1 - pi1
  }

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




