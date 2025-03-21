#%%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

#%%
class Config:
    '''
    Configure global settings for analysis.

    Note: Changing cv_folds and sensitivity_analysis alters the models and outputs produced.

    Attributes:
        file_location (str): Relative path to the dataset.
        file_path (str): Full path to the dataset file.
        cv_folds (int): Number of folds for cross-validation.
        run_eda (bool): Toggles running EDA.
        full_analysis (bool): Toggles running full-feature analysis.
        sensitivity_analysis (bool): Toggles running sensitivity analyses.

    '''
    def __init__(self):
        '''
        Initialise configuration with default settings.
        '''
        self.file_location = 'datasets/heart-disease.csv'
        self.file_path = os.path.join(os.path.dirname(__file__), self.file_location)
        self.cv_folds = 10  # Default to 10-fold cross-validation for our analysis
        self.run_eda = True              # Toggle for running the EDA pipeline
        self.full_analysis = True       # Toggle for full-feature-analysis
        self.sensitivity_analysis = True # Toggle for sensitivity analyses


# Instantiate configuration object
config = Config()

#%%
#---------------- Data and Preprocessing Functions ----------------#
def load_data():
    '''
    Load data from the dataset defined in config.
    '''
    df = pd.read_csv(config.file_path)
    return df

def display_dataset_info(df, target_col):
    '''
    Display dataset information.
    '''
    print("First five rows of the dataset:")
    print(df.head())
    print("\nDataset Information:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values in Each Column:")
    print(df.isnull().sum())

def preprocess_data(df):
    '''
    Preprocess data by converting categorical features into numeric.
    '''
    df['famhist'] = df['famhist'].replace({'Present': 1, 'Absent': 0})
    return df

def split_features_target(df, target_col):
    '''
    Split DataFrame into features and target.

    Returns:
        X: Feature matrix (DataFrame)
        Y: Target/response Series
        numeric_cols: List of numeric features excluding encoded categorical ones.
    '''
    Y = df[target_col]
    X = df.drop(columns=[target_col])
    # Exclude 'famhist' from numeric columns as it is a converted categorical variable.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.difference(['famhist'])
    return X, Y, numeric_cols

def visualise_data(df, X, Y, numeric_cols):
    '''
    Visualise data to understand distributions and relationships (EDA).
    '''
    sns.set(style="whitegrid")
    
    # Plot histograms of numeric features
    axes = X[numeric_cols].hist(figsize=(12, 10), bins=20)
    for ax in axes.flatten():
        ax.set_ylabel("Frequency")
    plt.suptitle("Histograms of Numeric Features", y=1.02)
    plt.show()
    
    # Plot boxplots of numeric features
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=X[numeric_cols], palette="Set3")
    plt.title("Boxplots of Numeric Features")
    plt.xticks(rotation=45)
    plt.show()
        
    # Plot pie chart for target distribution
    plt.figure(figsize=(6, 6))
    Y.value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90,
                          colors=sns.color_palette('pastel'))
    plt.title("Distribution of Coronary Heart Disease")
    plt.xlabel("(0 = No, 1 = Yes)")
    plt.ylabel('')
    plt.show()
    
    # Plot pair plot for numeric features coloured by target
    sns.pairplot(df.drop(columns='famhist'), hue="chd", diag_kind="hist")
    plt.suptitle("Pair Plot of Heart Disease Data", y=1.02)
    plt.show()
    
    # Plot pie chart for 'famhist'
    plt.figure(figsize=(6,6))
    df['famhist'].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                      startangle=90, colors=sns.color_palette('pastel'))
    plt.title("Distribution of Family History")
    plt.ylabel('')
    plt.show()
    
    # Plot correlation heatmap
    numeric_cols_df = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols_df].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # PCA Analysis (from here until end of this function)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X[numeric_cols])
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_scaled)
    
    # Display the explained variance
    print("Explained Variance Ratio:")
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"PC{i}: {ratio:.2%}")
    
    # Prepare the PCA results as a DataFrame (optional)
    pc_df = pd.DataFrame(data=principal_components, 
                         index=X.index, 
                         columns=['PC1', 'PC2'])
    
    # Display the PCA scores over the observations (if relevant)
    plt.figure(figsize=(10, 6))
    plt.scatter(pc_df['PC1'], pc_df['PC2'], c='grey', alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Scatter Plot of Numeric Features")
    plt.grid(True)
    plt.show()
    
    # Define a helper function to create a biplot
    def biplot(scores, coeff, labels):
        xs = scores[:, 0]
        ys = scores[:, 1]
        
        # Scale scores for better visualisation
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        
        plt.figure(figsize=(10, 7))
        plt.scatter(xs * scalex, ys * scaley, c='grey', alpha=0.5)
        
        for i in range(coeff.shape[0]):
            plt.arrow(0, 0, coeff[i, 0], coeff[i, 1],
                      color='r', width=0.002, head_width=0.02)
            plt.text(coeff[i, 0] * 1.1, coeff[i, 1] * 1.1,
                     labels[i], color='b', fontsize=10)
        
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Biplot of PCA on Numeric Features")
        plt.axis()
        plt.axhline(y=0, color='black', linewidth=0.8)
        plt.axvline(x=0, color='black', linewidth=0.8)
        plt.grid(False)
        plt.show()
    
    # Create and display the biplot using feature names as labels
    biplot(principal_components, pca.components_.T, labels=numeric_cols)

    

def split_and_scale_data(X, Y):
    '''
    Split data into training and test sets, and scale numeric features using StandardScaler.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, continuous_cols
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    continuous_cols = X.select_dtypes(include=[np.number]).columns.difference(['famhist'])
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    print("Training set shape:", X_train_scaled.shape)
    print("Test set shape:", X_test_scaled.shape)
    return X_train_scaled, X_test_scaled, y_train, y_test, continuous_cols

#---------------- Outlier Handling Function ----------------#
def remove_outliers_iqr(df, cols, multiplier=1.5):
    '''
    Remove outliers from data using InterQuartile Range.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (list): List of columns to check for outliers.
        multiplier (float): Multiplier for the IQR to define outlier thresholds.

    Returns:
        df_filtered: DataFrame with outliers removed.
    '''
    df_filtered = df.copy()
    for col in cols:
        if col in df_filtered.columns:
            Q1 = df_filtered[col].quantile(0.25)
            Q3 = df_filtered[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]
    df_filtered = df_filtered.reset_index(drop=True)
    return df_filtered

def remove_feature(df, feature_name):
    '''
    Remove a specified feature from data.
    '''
    df_modified = df.copy()
    if feature_name in df_modified.columns:
        df_modified = df_modified.drop(columns=[feature_name])
    return df_modified

#---------------- Wrapper Function for Model Evaluation ----------------#
def evaluate_model(model, param_grid, X_train, y_train, X_test, y_test, model_name='Model', extra_label='ROC', color=None):
    '''
    Evaluate a given model using GridSearchCV (if param_grid provided) and compute evaluation metrics.

    This function:
      - Runs grid search for hyperparameter tuning using CV folds defined in config.
      - Trains the best model.
      - Evaluates test data: accuracy, confusion matrix, ROC AUC, and plots the ROC curve.

    Args:
        model: A scikit-learn estimator.
        param_grid (dict): Hyperparameter grid for GridSearchCV. If empty or None, skip grid search.
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
        model_name (str): Name of the model (used in print statements).
        extra_label (str): Label used in the ROC plot.
        color: Optional; color for the ROC plot.

    Returns:
        best_estimator: Best model found (or fitted model if grid search was not performed).
        conf_matrix: Confusion matrix computed on test data.
        roc_auc_val: ROC AUC score.
    '''
    if param_grid:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=config.cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        print(f"{model_name} Best Parameters: {grid_search.best_params_}")
        print(f"{model_name} Best CV ROC AUC: {grid_search.best_score_:.4f}")
    else:
        best_estimator = model
        best_estimator.fit(X_train, y_train)
    
    y_pred = best_estimator.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    if hasattr(best_estimator, 'predict_proba'):
        y_prob = best_estimator.predict_proba(X_test)[:, 1]
        roc_auc_val = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
    else:
        y_scores = best_estimator.decision_function(X_test)
        roc_auc_val = roc_auc_score(y_test, y_scores)
        fpr, tpr, _ = roc_curve(y_test, y_scores)
    
    print(f"{model_name} Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"{model_name} Confusion Matrix:\n{conf_matrix}")
    print(f"{model_name} ROC AUC: {roc_auc_val:.4f}")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{extra_label} (AUC = {roc_auc_val:.2f})", color=color)
    plt.plot([0, 1], [0, 1], 'k--', label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.show()
    return best_estimator, conf_matrix, roc_auc_val

#---------------- Model-Specific Functions Using the Wrapper ----------------#
def run_logistic_regression_gridsearch(X_train, y_train, X_test, y_test):
    '''
    Run logistic regression with grid search over different penalty types and return a dictionary of results.
    '''
    results_by_penalty = {}
    param_grids = {
        'l2': {
            'penalty': ['l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        },
        'l1': {
            'penalty': ['l1'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear']
        },
        'none': {
            'penalty': [None],
            'solver': ['lbfgs', 'saga']
        }
    }
    for pen, grid in param_grids.items():
        print(f"\n============================")
        print(f"Run GridSearchCV for Logistic Regression ({pen.upper()})")
        model = LogisticRegression(max_iter=1000, random_state=42)
        best_model, conf_matrix, roc_auc_val = evaluate_model(
            model, grid, X_train, y_train, X_test, y_test,
            model_name=f"Logistic Regression ({pen.upper()})",
            extra_label="ROC"
        )
        results_by_penalty[pen] = {
            "best_model": best_model,
            "conf_matrix": conf_matrix,
            "roc_auc": roc_auc_val
        }
    return results_by_penalty

def run_LDA(X_train, y_train, X_test, y_test):
    '''
    Run LDA and return confusion matrix.
    '''
    model = LinearDiscriminantAnalysis()
    _, conf_matrix, _ = evaluate_model(
        model, None, X_train, y_train, X_test, y_test,
        model_name="LDA", extra_label="ROC"
    )
    return conf_matrix

def run_SVM(X_train, y_train, X_test, y_test):
    '''
    Run SVM with a linear kernel and return confusion matrix.
    '''
    model = SVC(kernel='linear', probability=True, random_state=42)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    _, conf_matrix, _ = evaluate_model(
        model, param_grid, X_train, y_train, X_test, y_test,
        model_name="SVM", extra_label="ROC"
    )
    return conf_matrix

def run_NB(X_train, y_train, X_test, y_test):
    '''
    Run Naïve Bayes and return confusion matrix.
    '''
    model = GaussianNB()
    _, conf_matrix, _ = evaluate_model(
        model, None, X_train, y_train, X_test, y_test,
        model_name="Naïve Bayes", extra_label="ROC"
    )
    return conf_matrix

def run_QDA(X_train, y_train, X_test, y_test):
    '''
    Run QDA and return confusion matrix.
    '''
    model = QuadraticDiscriminantAnalysis()
    _, conf_matrix, _ = evaluate_model(
        model, None, X_train, y_train, X_test, y_test,
        model_name="QDA", extra_label="ROC"
    )
    return conf_matrix

def run_decision_tree(X_train, y_train, X_test, y_test):
    '''
    Run decision tree with grid search and return confusion matrix.
    '''
    model = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10, 20],
        'ccp_alpha': [0.0, 0.001, 0.01, 0.1]
    }
    _, conf_matrix, _ = evaluate_model(
        model, param_grid, X_train, y_train, X_test, y_test,
        model_name="Decision Tree", extra_label="ROC"
    )
    return conf_matrix

def run_random_forest(X_train, y_train, X_test, y_test):
    '''
    Run Random Forest with grid search and return confusion matrix.
    '''
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'ccp_alpha': [0.0, 0.001, 0.01, 0.1],
        'max_features': ['sqrt', 'log2']
    }
    _, conf_matrix, _ = evaluate_model(
        model, param_grid, X_train, y_train, X_test, y_test,
        model_name="Random Forest", extra_label="ROC"
    )
    return conf_matrix

def run_ada_boost(X_train, y_train, X_test, y_test):
    '''
    Run AdaBoost with grid search and return confusion matrix.
    '''
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    model = AdaBoostClassifier(estimator=base_estimator, random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1, 10]
    }
    _, conf_matrix, _ = evaluate_model(
        model, param_grid, X_train, y_train, X_test, y_test,
        model_name="AdaBoost", extra_label="ROC"
    )
    return conf_matrix

def run_gradient_boosting(X_train, y_train, X_test, y_test):
    '''
    Run Gradient Boosting with grid search and return confusion matrix.
    '''
    model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2, 1],
        'max_depth': [3, 5, 7, 10]
    }
    _, conf_matrix, _ = evaluate_model(
        model, param_grid, X_train, y_train, X_test, y_test,
        model_name="Gradient Boosting", extra_label="ROC"
    )
    return conf_matrix

def run_knn_one(X_train, y_train, X_test, y_test):
    '''
    Run k-Nearest Neighbours with k=1 and return confusion matrix.
    '''
    model = KNeighborsClassifier(n_neighbors=1)
    _, conf_matrix, _ = evaluate_model(
        model, None, X_train, y_train, X_test, y_test,
        model_name="kNN (k=1)", extra_label="ROC"
    )
    return conf_matrix

def run_knn_cv(X_train, y_train, X_test, y_test):
    '''
    Run k-Nearest Neighbours with k chosen via cross-validation and return confusion matrix.
    '''
    model = KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(1, 21))}  # Test k from 1 to 20
    _, conf_matrix, _ = evaluate_model(
        model, param_grid, X_train, y_train, X_test, y_test,
        model_name="kNN (CV)", extra_label="ROC"
    )
    return conf_matrix

def print_confusion_matrices(conf_matrices):
    '''
    Print the confusion matrices for all models.
    '''
    print(f'\nConfusion Matrix for Logistic Regression:\n{conf_matrices.get("logistic", "Not available")}')
    print(f'\nConfusion Matrix for SVM:\n{conf_matrices.get("svm", "Not available")}')
    print(f'\nConfusion Matrix for LDA:\n{conf_matrices.get("lda", "Not available")}')
    print(f'\nConfusion Matrix for Naïve Bayes:\n{conf_matrices.get("nb", "Not available")}')
    print(f'\nConfusion Matrix for QDA:\n{conf_matrices.get("qda", "Not available")}')
    print(f'\nConfusion Matrix for Decision Tree:\n{conf_matrices.get("tree", "Not available")}')
    print(f'\nConfusion Matrix for Random Forest:\n{conf_matrices.get("rf", "Not available")}')
    print(f'\nConfusion Matrix for AdaBoost:\n{conf_matrices.get("adaboost", "Not available")}')
    print(f'\nConfusion Matrix for Gradient Boosting:\n{conf_matrices.get("gradient_boosting", "Not available")}')
    print(f'\nConfusion Matrix for kNN (k=1):\n{conf_matrices.get("knn_one", "Not available")}')
    print(f'\nConfusion Matrix for kNN (CV):\n{conf_matrices.get("knn_cv", "Not available")}')

#---------------- Pipeline Functions ----------------#
def run_eda_pipeline(df):
    '''
    Run the EDA pipeline: display dataset information and visualise data.
    '''
    target_col = 'chd'
    X, Y, numeric_cols = split_features_target(df, target_col)
    display_dataset_info(df, target_col)
    visualise_data(df, X, Y, numeric_cols)

def run_model_pipeline(df, pipeline_label='Pipeline with All Features'):
    '''
    Run the model pipeline: split data, train models, and evaluate them.
    Note: This pipeline does NOT perform EDA.
    '''
    print(f"\n--- Running {pipeline_label} ---")
    target_col = 'chd'
    X, Y, _ = split_features_target(df, target_col)
    X_train_scaled, X_test_scaled, y_train, y_test, _ = split_and_scale_data(X, Y)
    
    # Run logistic regression grid search (for all penalties)
    logistic_results = run_logistic_regression_gridsearch(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Initialise a dictionary for confusion matrices
    conf_matrices = {}
    conf_matrices['logistic'] = logistic_results['l2']['conf_matrix']
    conf_matrices['lda'] = run_LDA(X_train_scaled, y_train, X_test_scaled, y_test)
    conf_matrices['svm'] = run_SVM(X_train_scaled, y_train, X_test_scaled, y_test)
    conf_matrices['nb'] = run_NB(X_train_scaled, y_train, X_test_scaled, y_test)
    conf_matrices['qda'] = run_QDA(X_train_scaled, y_train, X_test_scaled, y_test)
    conf_matrices['tree'] = run_decision_tree(X_train_scaled, y_train, X_test_scaled, y_test)
    conf_matrices['rf'] = run_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    conf_matrices['adaboost'] = run_ada_boost(X_train_scaled, y_train, X_test_scaled, y_test)
    conf_matrices['gradient_boosting'] = run_gradient_boosting(X_train_scaled, y_train, X_test_scaled, y_test)
    conf_matrices['knn_one'] = run_knn_one(X_train_scaled, y_train, X_test_scaled, y_test)
    conf_matrices['knn_cv'] = run_knn_cv(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print(f"\n=== All Confusion Matrices for {pipeline_label} ===")
    print_confusion_matrices(conf_matrices)
    return logistic_results, conf_matrices

#%%
#---------------- Main Execution Function ----------------#
def main():
    '''
    Run the complete analysis: load and preprocess data, run the EDA pipeline (if enabled),
    run the model pipeline (if enabled), and finally perform sensitivity analyses (if enabled).
    '''
    df = load_data()
    df = preprocess_data(df)
    
    # Run EDA pipeline if toggled on in config
    if config.run_eda:
        print("\n================== Running EDA Pipeline ==================")
        run_eda_pipeline(df)
    
    # Run model pipeline if toggled on in config
    if config.full_analysis:
        print("\n================== Running Full-Feature Pipeline ==================")
        logistic_results_all, conf_matrices_all = run_model_pipeline(df, pipeline_label='Pipeline with All Features')
    
    if config.sensitivity_analysis:
        # Sensitivity analysis: remove alcohol feature
        print("\n================== Running Sensitivity Analysis: Pipeline Without 'alcohol' Feature ==================")
        df_no_alcohol = remove_feature(df, 'alcohol')
        _ = run_model_pipeline(df_no_alcohol, pipeline_label="Pipeline without 'alcohol'")
        
        # Sensitivity analysis: remove outliers using IQR
        print("\n================== Running Sensitivity Analysis: Pipeline With Outliers Removed (IQR Method) ==================")
        outlier_cols = ['alcohol', 'ldl', 'obesity', 'tobacco', 'typea']
        df_no_outliers = remove_outliers_iqr(df, outlier_cols)
        _ = run_model_pipeline(df_no_outliers, pipeline_label="Pipeline with Outliers Removed")
    
    print('\n================== Analysis Completed ==================')

if __name__ == '__main__':
    main()

# %%
