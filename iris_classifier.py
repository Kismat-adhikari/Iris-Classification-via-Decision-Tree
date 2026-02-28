import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

# === LOAD AND EXPLORE DATASET ===
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nClass Distribution:")
print(df['species'].value_counts())
print(f"\nBasic Statistics:")
print(df.describe())
print(f"\nMissing Values:")
print(df.isnull().sum())


# === PREPROCESSING ===
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("=== PREPROCESSING ===")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('heatmap.png')
plt.show()
print("Heatmap saved as heatmap.png")



# === MODEL TRAINING ===
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("=== MODEL TRAINING ===")
print("Model trained successfully!")
print(f"Tree Depth: {model.get_depth()}")
print(f"Number of Leaves: {model.get_n_leaves()}")

# Visualize the tree
plt.figure(figsize=(15,8))
plot_tree(model, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True)
plt.title('Decision Tree Visualization')
plt.tight_layout()
plt.savefig('decision_tree.png')
plt.show()
print("Decision tree saved as decision_tree.png")


# === EVALUATION ===
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("=== MODEL EVALUATION ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix saved as confusion_matrix.png")


# === CLI INTERFACE ===
def cli():
    print("\n" + "="*50)
    print("  Iris Flower Species Classifier")
    print("="*50)
    print("Enter flower measurements to predict species")
    print("Type 'quit' to exit\n")
    
    while True:
        print("-"*50)
        try:
            sepal_length = input("Sepal Length (cm) [e.g. 5.1]: ")
            if sepal_length.lower() == 'quit':
                print("Goodbye!")
                break
                
            sepal_length = float(sepal_length)
            sepal_width = float(input("Sepal Width  (cm) [e.g. 3.5]: "))
            petal_length = float(input("Petal Length (cm) [e.g. 1.4]: "))
            petal_width = float(input("Petal Width  (cm) [e.g. 0.2]: "))
            
            user_input = np.array([[sepal_length, sepal_width, 
                                    petal_length, petal_width]])
            user_input_scaled = scaler.transform(user_input)
            prediction = model.predict(user_input_scaled)
            species = iris.target_names[prediction[0]]
            
            print(f"\n🌸 Predicted Species: {species.upper()}")
            print(f"(Confidence based on Decision Tree depth 3 model)\n")
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values only.\n")

cli()