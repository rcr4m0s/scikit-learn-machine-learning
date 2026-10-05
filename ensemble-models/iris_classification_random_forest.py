import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def run_classification_pipeline():
    print("=== Scikit-Learn Classification Pipeline ===")
    
    
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    
  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
   
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
 
    y_pred = model.predict(X_test_scaled)
    
 
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%\n")
    
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    run_classification_pipeline()