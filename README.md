# Machine_Learning_with_R
___

# Machine Learning Methods with R

This repository presents various machine learning techniques using R. 

## Table of Contents
1. [Least Squares (LS)](#least-squares-ls)
2. [PCR (Principal Component Regression)](#pcr-principal-component-regression)
3. [PLS Regression (Partial Least Squares Regression)](#pls-regression-partial-least-squares-regression)
4. [Shrinkage Methods](#shrinkage-methods)
5. [Linear Methods for Classification](#Linear-Methods-for-Classification)
6. [Nonlinear Methods](#Nonlinear-Methods)
7. [Tree-Based Methods](#Tree-Based-Methods)
8. [Support Vector Machines (SVM)](#Support-Vector-Machines-(SVM))

---

## Least Squares (LS)

- **Generation of the Data:** Synthetic data generated and relationships visualized.
- **Model Training:** Three linear regression models were trained.
- **Model Comparison:** Models were compared using various functions.
- **Body Fat Data Analysis:** The `fat` dataset was analyzed and linear regression models built and evaluated.

---

## PCR (Principal Component Regression)

PCR is a regression technique that first reduces the predictors using PCA and then builds a regression model based on the principal components.

- Built a PCR model and evaluated its performance.
- Plotted MSEP vs. Number of Components.

---

## PLS Regression (Partial Least Squares Regression)

PLS regression is useful when predictors are highly collinear or there are more predictors than observations.

- Built a PLS regression model and evaluated its performance.
- Plotted MSEP vs. Number of Components.

---

## Shrinkage Methods

Shrinkage methods are regression techniques that include some form of regularization to avoid overfitting and handle multicollinearity.

### Ridge Regression

Ridge regression shrinks the coefficients towards zero using an L2 penalty. 

- A ridge regression model was fitted for various values of the penalty term `lambda`.
- The model's performance was visualized using Generalized Cross-Validation (GCV) against `lambda`.
- After selecting the optimal `lambda`, predictions were generated and visualized against actual values.

### Lasso Regression

Lasso regression uses an L1 penalty which can shrink coefficients exactly to zero, effectively performing feature selection.

- The lasso regression model was fitted.
- The path of coefficients was visualized as `lambda` changes.
- Cross-validation was conducted to select the best value for `lambda`.
- Predictions were made using the selected model and visualized against actual values.

---

## Linear Methods for Classification

Linear methods for classification aim to find a linear boundary between classes. 

### Data Preparation and Visualization

The Pima Indians Diabetes dataset is loaded and visualized. Following this, any missing values are omitted from the dataset. To prepare the data for a linear regression approach, the binary outcome (diabetes positive/negative) is encoded into two separate columns.

### Linear Regression for Classification

The dataset is split into training and testing sets, after which a linear regression model is trained on the training set. The model's predictions on the test set are then used to compute the misclassification rate.

### Classical LDA (Linear Discriminant Analysis)

Linear Discriminant Analysis (LDA) is used to classify the data. LDA aims to maximize the distance between the means of the two classes while minimizing the spread (variance) within each class. After cross-validation and model building for LDA, the model's performance is assessed with a misclassification rate.

### QDA (Quadratic Discriminant Analysis)

Quadratic Discriminant Analysis (QDA) is also applied to the data. QDA, like LDA, uses Bayes' theorem but assumes each class has its covariance matrix. A QDA model is trained and its predictions on the test set are used to compute the misclassification rate.

### Regularized Discriminant Analysis

Regularized Discriminant Analysis, a compromise between LDA and QDA, is used. It strikes a balance between assuming a shared covariance matrix (like LDA) and individual covariance matrices for each class (like QDA). Using the `rda` function from the `klaR` package, a model is trained and its performance is assessed using a misclassification rate.

### Logistic Regression

Logistic Regression is employed to estimate the probability that a given instance belongs to a particular category. A logistic regression model is trained on the data, followed by a stepwise feature selection process using the Akaike information criterion (AIC). Predictions made on the test set are then visualized. Subsequently, cross-validation is conducted for the logistic regression model, and its performance is assessed using the misclassification rate.

---

## Nonlinear Methods

**Dataset:** The dataset `wtloss` from the `MASS` package tracks weight loss over a number of days.

### Polynomial Regression:
- The weight loss data is plotted.
- A linear regression model (`lm1`) is fit to the data.
- A quadratic regression model (`lm2`) is also fit.
- The linear and quadratic regression models are visualized.

### Nonlinear Regression using Exponential Decay:
- An exponential decay model is fit to the data using `nls`. This nonlinear regression model tries to capture the natural decay pattern that might be seen in weight loss.
- The fitted curve is plotted.
- An optimization approach (`optim`) is used to fit the same model.

### Interpolation with Splines:
- The `lecturespl` function is introduced to generate basis functions for splines, based on the input data `x`.
- The `x` and `y` data are interpolated using splines with 2 and 6 knots respectively.
- The results are visualized.

### Built-in Spline Functions:
- The `bs` function from the `splines` package is used to demonstrate how splines can be fit and visualized using built-in functions.
- Different spline fits are visualized based on specified degrees of freedom.

### Smoothing Splines:
- The `smooth.spline` function is used to fit the data with a smooth curve, both with a fixed degree of freedom and through cross-validation (`cv=TRUE`).
- The results are visualized.

### Generalized Additive Models (GAMs):
- The `mgcv` package is introduced, which provides the `gam` function for Generalized Additive Models.
- A GAM is fit to the `sin` curve data. This model is more flexible and can adapt to the underlying pattern in the data.
- The fitted GAM is visualized with standard errors.
- Another GAM is introduced for diabetes classification, using the `pid` dataset. This GAM acts as a logistic regression model but uses smoothed versions of predictors.

**Key Takeaways:**
1. **Polynomial Regression:** Captures curvilinear trends in the data.
2. **Exponential Decay:** Models data that decreases (or increases) at a rate proportional to its current value.
3. **Splines:** Allow for flexible data modeling by dividing the data into sections and fitting polynomials.
4. **Smoothing Splines:** Offer a method to fit data smoothly while controlling for overfitting.
5. **GAMs:** Extend linear models by permitting nonlinear relationships through smooth functions.

---

## Tree-Based Methods

Tree-based methods construct decision trees for predictions, making them suitable for both regression and classification problems. The primary advantage is that they can model non-linear relationships and offer a graphical representation, aiding interpretability.

### Regression Trees

Using the `fat` dataset from the "UsingR" library, this section details constructing a decision tree to predict body fat percentage based on various attributes.

**1. Data Preprocessing:**
   - Data is fetched and unwanted rows and columns are discarded. 
   - The dataset is divided into training and test sets.
 
**2. Decision Tree Model:**
   - A decision tree model is trained using the `rpart` package.
   - The trained model is visualized.
   - Predictions are made on test data, and the RMSE is calculated for evaluation.
   - Overfitting is checked via a complexity parameter plot, and the tree is pruned if required.

### Classification Trees

The aim in this section is to classify whether a bank client would subscribe to a term deposit or not.

**1. Data Loading & Preprocessing:**
   - Data is loaded, cleaned, and processed.
   - The dataset is partitioned into training and test subsets.

**2. Classification Tree Model:**
   - A classification tree model is created.
   - The tree's performance is gauged by comparing its predictions against actual values.

### Random Forests

Random Forest is an ensemble of decision trees. By training multiple trees and aggregating their predictions, Random Forests can provide improved accuracy and prevent overfitting.

**1. Basic Random Forest:**
   - A Random Forest model is constructed and trained.
   - The importance of each variable in making predictions is plotted.
   - The Random Forest's performance is measured against the test data.

**2. Balancing Classes in Random Forest:**
   - Given an imbalanced dataset, multiple methods are applied to balance the classes, ensuring that one class doesn't dominate the predictions.
   - The effectiveness of each method is evaluated.

In conclusion, tree-based techniques, from basic decision trees to the ensemble method of Random Forests, offer powerful tools for data analysis. They're particularly notable for their ability to model complex, non-linear relationships and provide insights into the data.

---

## Support Vector Machines (SVM)

Support Vector Machines (SVM) are powerful algorithms used for classification and regression tasks. The core principle behind SVM is to find the hyperplane that best separates different classes in the feature space.

### Basic SVM Example
In this example, a two-dimensional dataset with two classes is initially generated. Then, a linear SVM model is trained to classify these points. The model is visualized, highlighting the support vectors—data points that are essential in determining the hyperplane's position. The SVM's effectiveness is demonstrated by its ability to classify most of the data correctly, with only minimal misclassification.

### More SVM Examples
A more complex dataset is used, where points from two classes are distributed across a larger area. Two SVM models with radial basis function (RBF) kernels are trained. The first model uses a specific gamma and cost, while the second employs a higher cost to illustrate the effect of the regularization parameter. The models are visualized, and their performances compared.

### Classification Example
In the classification example, the `pid` dataset is utilized. The dataset is split into training and test sets. An SVM with an RBF kernel is trained on the training set. Predictions are made on the test set, and the misclassification rate is computed. Furthermore, a tuning process is conducted to find the optimal gamma and cost values for the SVM. The model is then retrained using the optimal parameters and evaluated again on the test set.

### Regression Example
For the regression task, the `fat` dataset from the `UsingR` package is leveraged. After some preprocessing, the dataset is split into training and test sets. An SVM regression model with an RBF kernel is trained on the training set. Predictions are then made on the test set, and the root mean square error (RMSE) is calculated to evaluate the model's performance.

In summary, Support Vector Machines offer a robust method to handle both classification and regression problems. Their hyperparameters, such as cost and kernel type, can significantly influence the model's performance, making tuning essential.



