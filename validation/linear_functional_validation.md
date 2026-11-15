# Validation of the general linear-functional API

Validated on November 15, 2026, against implementation commit
[`b75b51aa8a028a5b4695737465fe7e5164db3bfc`](https://github.com/Larsvanderlaan/calibratedDML/commit/b75b51aa8a028a5b4695737465fe7e5164db3bfc).
The new tests use the same deterministic CSV in R and Python. Their reference
calculations do not call the package's PAVA, binning, or prediction helpers.

The tested three-input calculations are correct in both implementations. No
estimator or inference algorithm was changed. The changes add regression tests,
a direct R/Python comparison, continuous-integration coverage, and a precise
description of the step-function class used for calibration.

## 1. A finite population with three evaluation inputs

Let X be uniform on {0, 1}, let A conditional on X be uniform on {0, 1, 2}, and
let epsilon be independent and uniform on {-1, -1/2, 0, 1/2, 1}. Set

\[
Y=1+X+2A+\epsilon,\qquad w(X)=1+X.
\]

The fixture enumerates all 30 combinations, in order of X, A, and epsilon.
The three evaluation inputs are (0, X), (1, X), and (2, X). Define

\[
c(0)=(-2,3,-1),\qquad c(1)=(-1,1,2),\qquad
m_c(O;h)=\sum_{a=0}^2 c_a(X)h(a,X).
\]

These coefficients are observation-specific and signed. Their row sums vary
between 0 and 2, so the example is not restricted to a contrast with zero total
coefficient. Write

\[
P_w h=\frac{E[w(X)h]}{E[w(X)]}.
\]

The regression and Riesz representer under this weighted law are

\[
\mu_0(a,x)=1+x+2a,\qquad \alpha_0(a,x)=3c_a(x).
\]

Indeed, for every function h on the six-point (A, X) domain,

\[
P_w\{\alpha_0(A,X)h(A,X)\}
=P_w\left\{\sum_{a=0}^2\tfrac13\,3c_a(X)h(a,X)\right\}
=P_w m_c(O;h).
\]

The conditional functional values are 2 at X = 0 and 14 at X = 1. Thus

\[
\psi_0=P_w m_c(O;\mu_0)=\frac{2+2\cdot14}{3}=10.
\]

For the main numerical check, supply the deliberately misspecified functions

\[
\widehat\mu_{\rm eval}=X+(0,2,1),\qquad
\widehat\alpha_{\rm eval}=(-2,0,2),
\]

and select column A for each observed-input prediction. These are fixed
functions supplied to the API; no nuisance learner is trained in this example.
The CSV stores all outcomes, weights, coefficients, and observed/evaluation
predictions explicitly. Its weights sum to 45.

## 2. Uncalibrated AIPW identity

The uncalibrated row score is

\[
s_i=\sum_{a=0}^2c_a(X_i)\widehat\mu(a,X_i)
 +\widehat\alpha(A_i,X_i)\{Y_i-\widehat\mu(A_i,X_i)\}.
\]

For arbitrary fixed supplied functions mu and alpha, conditioning on (A, X)
and using the Riesz identity gives

\[
\begin{aligned}
P_w\{m_c(O;\mu)+\alpha(Y-\mu)\}-\psi_0
&=P_w\{m_c(O;\mu-\mu_0)+\alpha(\mu_0-\mu)\}\\
&=P_w\{(\alpha-\alpha_0)(\mu_0-\mu)\}.
\end{aligned}
\]

The tests recover exactly 10 with either the true regression or the true
representer. In the main misspecified case, the exact scores are:

| X | A = 0 | A = 1 | A = 2 |
|---|---|---|---|
| 0 | 3 - 2 epsilon | 5 | 13 + 2 epsilon |
| 1 | 4 - 2 epsilon | 6 | 14 + 2 epsilon |

The weighted plug-in mean is 17/3, and the weighted correction is 2. Therefore

\[
\widehat\psi_{\rm raw}=23/3=7.666666666666667.
\]

Both APIs are checked against every row score, not only this final mean.

## 3. Independent solutions of both calibration problems

For either nuisance, write the objective at sorted observed-prediction knots
as

\[
\min_{u_1\le\cdots\le u_K}
\sum_{\ell=1}^K(A_\ell u_\ell^2-2B_\ell u_\ell).
\]

The tests independently evaluate the interval-average characterization

\[
u_j=\max_{a\le j}\min_{b\ge j}
\frac{\sum_{\ell=a}^b B_\ell}{\sum_{\ell=a}^b A_\ell}.
\]

This calculation uses direct interval sums, without the production PAVA
implementation. Evaluation bins are also constructed independently by
counting knot comparisons.

For regression calibration,
\(A_\ell=\sum_iw_i I(\widehat\mu_i=t_\ell)\) and
\(B_\ell=\sum_iw_iY_i I(\widehat\mu_i=t_\ell)\).

| Regression knot | A | B | B/A | Fitted value |
|---:|---:|---:|---:|---:|
| 0 | 5 | 5 | 1 | 1 |
| 1 | 15 | 45 | 3 | 3 |
| 2 | 15 | 75 | 5 | 23/5 |
| 3 | 10 | 40 | 4 | 23/5 |

For Riesz calibration, A again uses observed predictions. The linear mass B
uses all three functional evaluation columns, with their signs:

\[
B_\ell=\sum_{i=1}^{30}\sum_{a=0}^2
w_i c_a(X_i)
 I\{\widehat\alpha^{\rm eval}_{ia}\text{ is in step interval }\ell\}.
\]

| Riesz knot | A | B | B/A | Fitted value |
|---:|---:|---:|---:|---:|
| -2 | 15 | -60 | -4 | -4 |
| 0 | 15 | 75 | 5 | 4 |
| 2 | 15 | 45 | 3 | 4 |

Thus both fits require pooling. The regression fit pools knots 2 and 3; the
Riesz fit pools knots 0 and 2.

### Both empirical calibration equations

Let f and g be the fitted regression and Riesz maps, and write
\(\widetilde\mu_i=f(\widehat\mu_i)\) and
\(\widetilde\alpha_i=g(\widehat\alpha_i)\).
For every function h on the corresponding set of fitted levels,

\[
\sum_i w_i h(\widetilde\mu_i)(Y_i-\widetilde\mu_i)=0,
\]

and

\[
\sum_i w_i\left[
\widetilde\alpha_i h(\widetilde\alpha_i)
-\sum_{a=0}^2c_a(X_i)
 h\{g(\widehat\alpha^{\rm eval}_{ia})\}
\right]=0.
\]

The tests take h to be every fitted-level indicator, which establishes each
identity for all functions on those levels. They also check the KKT conditions.
With

\[
S_\ell=\sum_{k\le\ell}(B_k-A_ku_k),
\]

these conditions are

\[
S_\ell\ge0,\qquad S_K=0,\qquad
S_\ell(u_{\ell+1}-u_\ell)=0\quad(\ell<K).
\]

The prefix residuals are (0, 0, 6, 0) for regression and (0, 15, 0) for Riesz.
They need not vanish at every original knot. They vanish at fitted-block
boundaries, giving zero total residual within each fitted-level block.

The calibrated functional contributions are 44/5 at X = 0 and 54/5 at X = 1.
The weighted plug-in mean is 152/15 and the weighted residual correction is
16/9. Hence

\[
\widehat\psi_{\rm calibrated}=536/45=11.911111111111111.
\]

The different regression and Riesz fitted-level partitions do not force the
correction to vanish; its value here is 16/9. The exact calibrated scores are:

| X | A = 0 | A = 1 | A = 2 |
|---|---|---|---|
| 0 | 44/5 - 4 epsilon | 12/5 + 4 epsilon | 84/5 + 4 epsilon |
| 1 | 74/5 - 4 epsilon | 42/5 + 4 epsilon | 82/5 + 4 epsilon |

Both initial nuisance functions are misspecified here, so neither the raw nor
the calibrated estimate is required to equal the population target 10.

### Step-function class

Each calibrator is nondecreasing and right-continuous, with knots at distinct
positive-weight observed predictions and constant extension at both tails.
Prediction values found only in evaluation matrices do not create new knots.
The updated R and Python documentation makes this restriction explicit.

This restriction matters for the Riesz loss. If a functional evaluation occurs
above all observed scores with a positive coefficient, an unrestricted
monotone function could increase at that evaluation-only point without a
corresponding observed quadratic penalty. The unrestricted problem can then
be unbounded below. The implemented constant-tail class avoids that problem.
The tests also cover evaluation points between knots, both tails, and
zero-weight observations.

## 4. Wald and fixed-nuisance jackknife calculations

For the optional Wald calculation, the implemented variance is

\[
\widehat{\operatorname{se}}^2
=\sum_i\left\{\frac{w_i}{45}(s_i-\widehat\psi)\right\}^2.
\]

| Method | Exact variance | Standard error |
|---|---:|---:|
| Uncalibrated | 908/1215 | 0.864479671756750 |
| Calibrated | 274952/273375 | 1.002880168729226 |

For deletion group v, subset outcomes, weights, observed predictions,
evaluation matrices, and coefficient rows together. Refit f and g on that
subset and recompute the final weighted mean. Initial nuisance predictions
and functional coefficients remain fixed.

For V groups, the reference calculation is

\[
z_v=V\widehat\psi-(V-1)\widehat\psi_{-v},\qquad
\widehat{\operatorname{se}}^2
=\frac{1}{V(V-1)}\sum_v(z_v-\bar z)^2
=\frac{V-1}{V}\sum_v
 (\widehat\psi_{-v}-\overline{\widehat\psi_{-}})^2.
\]

The default V = 10 deletes three rows per group in this fixture and uses a
t distribution with nine degrees of freedom. The interval is centered at the
full estimate, not the average pseudo-value.

Both language tests independently solve the two isotonic problems after every
deletion. They check every leaveout estimate, pseudo-value, standard error,
degree of freedom, and confidence limit. They also demonstrate that the
leaveout estimates differ from averages of frozen full-sample calibrated
scores. Callable coefficients are extracted using six calls on the original
30-by-3 matrix (zero, three basis columns, and two validation probes); the
callback is not called on deletion samples. Inputs are checked for mutation.

The leave-one-out reference has V = 30. Its calibrated standard error is
1.3783388993802286, and its t29 interval is
[9.092091537194102, 14.73013068502812].

## 5. Direct R/Python agreement

The standalone parity driver compares native stored outputs, including
pseudo-values, standard errors, degrees of freedom, and confidence limits.
For a native ten-fold partition from one language, the other language refits
the same fixed inputs on exactly those deletion groups. This is done in both
directions. Leave-one-out outputs are aligned by deleted observation.

The seed is 7 in both languages, but their generators produce different
partitions. The standard errors differ across those partitions as expected:

| Partition or calculation | Uncalibrated SE | Calibrated SE |
|---|---:|---:|
| Python's ten-fold partition, reproduced in R | 0.768547444332364 | 1.20376418982384 |
| R's ten-fold partition, reproduced in Python | 0.634631208858802 | 1.35134955012023 |
| Leave-one-out, identical deletions | 0.885422352087266 | 1.37833889938023 |

There are 356 comparisons for the uncalibrated method and 370 for the
calibrated method: **726 values in total**. The maximum absolute differences
are 6.39e-14 and 1.40e-13, respectively, below the absolute tolerance 1e-10.
Compared fields include coefficient matrices, fitted maps, calibrated
predictions, functional values, scores, Wald output, and jackknife output.

## 6. Reproduction and test results

Install the Python package and testing dependency, and install the R package
and its declared dependencies. From the repository root:

```sh
python -m pip install -e '.[dev]'
R CMD INSTALL .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/python/test_linear_functional.py tests/python/test_vfold_jackknife.py
Rscript -e 'library(calibratedDML); testthat::test_file("tests/testthat/test-linear-functional.R", stop_on_failure = TRUE)'
Rscript -e 'library(calibratedDML); testthat::test_file("tests/testthat/test-vfold-jackknife.R", stop_on_failure = TRUE)'
python validation/check_linear_functional_parity.py
```

The parity driver also accepts `--rscript /path/to/Rscript`. Disabling Python
plugin autoload makes the focused run independent of unrelated pytest plugins
installed in the calling environment; it is not required by the package.

| Focused check | Before | After | Result |
|---|---:|---:|---|
| Python linear-functional and jackknife files | 16 tests | 27 tests | All pass |
| R linear-functional file | 7 tests / 62 expectations | 13 tests / 133 expectations | All pass |
| R jackknife file | 5 tests / 22 expectations | 5 tests / 22 expectations | All pass |
| Direct R/Python comparison | Not present | 726 values | All agree |

The final focused runs have no failures, errors, warnings, or skips.
The runtime versions used were Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
pandas 2.2.3, R 4.5.0, and testthat 3.2.3.

## 7. Implementation findings and changes

1. **No numerical defect was found in the tested general-functional path.**
   Signed row-specific coefficients are correctly used in both the
   functional evaluation and Riesz linear masses. R's column-major and
   Python's row-major flattening each keep coefficients and evaluation bins
   aligned.
2. **The Python CI workflow omitted both relevant test files.** Its test
   command now includes `test_linear_functional.py` and
   `test_vfold_jackknife.py`. Both workflows respond to changes in shared
   fixtures. The R workflow also runs the direct parity driver after
   installing the current Python package, and responds to relevant Python
   source and parity-script changes.
3. **The documented optimization class needed precision.** The R source,
   R reference page, and Python docstring now state the observed-knot,
   constant-tail step-function restriction.
4. **Validation is now reproducible.** The shared CSV, independent reference
   tests, parity driver, and this note are included in the repository.

These deterministic checks establish the stated identities and numerical
calculations on the tested inputs. They do not establish confidence-interval
coverage or asymptotic validity for every user-supplied functional. No nuisance
training, default setting, or estimator behavior was changed.
