\name{calibrated_linear_functional}
\alias{calibrated_linear_functional}
\title{Calibrated DML for a Linear Functional}
\usage{
calibrated_linear_functional(
  Y, regression, riesz, functional,
  regression_evaluations = NULL, riesz_evaluations = NULL,
  sample_weight = NULL, calibration_method = c("isotonic", "none"),
  inference = c("jackknife", "wald", "bootstrap"), jackknife_folds = 10,
  bootstrap_reps = 200, conf_level = 0.95, seed = NULL,
  name = "linear_functional"
)
}
\arguments{
\item{Y}{Outcome vector.}
\item{regression}{Cross-fitted regression predictions at observed inputs.}
\item{riesz}{Cross-fitted Riesz predictions at observed inputs.}
\item{functional}{A row-wise linear function from an n-by-k prediction matrix to n contributions, a length-k coefficient vector, or an n-by-k coefficient matrix.}
\item{regression_evaluations}{Regression predictions at every input used by the functional, with one row per observation. Defaults to observed predictions.}
\item{riesz_evaluations}{Riesz predictions at those same inputs, in the same column order. Defaults to observed predictions.}
\item{sample_weight}{Optional non-negative observation weights.}
\item{calibration_method}{Isotonic PAVA calibration, or none.}
\item{inference}{V-fold jackknife (default), Wald, or bootstrap.}
\item{jackknife_folds}{Number V of deletion groups, default 10. Intervals use Student-t critical values with V - 1 degrees of freedom.}
\item{bootstrap_reps}{Number of bootstrap replicates, default 200.}
\item{conf_level}{Confidence level, default 0.95.}
\item{seed}{Optional random seed.}
\item{name}{Label for the functional.}
}
\description{Estimates E[sum_j c_j(O) * mu(T_j(O))] from cross-fitted regression and Riesz estimates. PAVA minimizes squared-error loss for the regression and the quadratic Riesz loss for the representer, allowing signed representers.}
\details{Both observed-input predictions and predictions at the functional's inputs must use the appropriate cross-fitted nuisance estimates. For an ATE, evaluation columns contain predictions under control and treatment. Use the same column order for the regression and Riesz evaluations.

Each calibrator is a nondecreasing, right-continuous step function with knots at the distinct positive-weight observed predictions and constant extension at both tails. The Riesz loss is minimized over this class; prediction values seen only in the evaluation matrix do not add knots.

The default jackknife holds all initial predictions and coefficients fixed while refitting calibration and recomputing means.}
\value{An object with summary and confint methods, calibrated predictions, calibration maps, and resampling diagnostics.}
\references{van der Laan, Luedtke, and Carone, Doubly robust inference via calibration, arXiv:2411.02771.}
