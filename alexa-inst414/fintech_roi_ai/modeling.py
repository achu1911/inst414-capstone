# Modeling

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Logistic Regression --> my baseline model
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        solver="liblinear", # Used liblinear because it's fast and works well for smaller + high-dimensional datasets  
        penalty="l2",
        class_weight="balanced", # handles the 26.5% churn imbalance
        max_iter=200 # increased max iter so model can converge iwth many dummies
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1       
    )
    model.fit(X_train, y_train)
    return model

def train_gradient_boost(X_train, y_train):
    model = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=100,   # reduce trees
        max_depth=2        # smaller trees = MUCH faster
    )
    model.fit(X_train, y_train)
    return model
