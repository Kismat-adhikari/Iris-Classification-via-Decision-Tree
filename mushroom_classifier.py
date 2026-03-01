import pandas as pd

df = pd.read_csv('mushrooms.csv')

print("Dataset Shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nClass Distribution:")
print(df['class'].value_counts())

print("\nMissing Values:", df.isnull().sum().sum())


from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()

le = LabelEncoder()
for col in df_encoded.columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

print("After Encoding - First 5 Rows:")
print(df_encoded.head())

print("\nFeatures Shape:", X.shape)

print("\nTarget Distribution:")
print(y.value_counts())
print("0 = edible, 1 = poisonous")




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)



from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print("Model trained successfully!")
print("Tree Depth:", model.get_depth())
print("Number of Leaves:", model.get_n_leaves())





from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['edible', 'poisonous']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))





import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['edible', 'poisonous'], yticklabels=['edible', 'poisonous'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print("Confusion matrix saved!")





from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns.tolist(), class_names=['edible', 'poisonous'], filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.savefig('decision_tree.png')
plt.show()

print("Decision tree saved!")




feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print("Feature Importances:")
print(feature_importance)

plt.figure(figsize=(10,6))
feature_importance.plot(kind='bar', color='steelblue')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("Feature importance plot saved!")








from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5)

print("Cross Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())










from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross Validation Score:", grid_search.best_score_)





def predict_mushroom():
    print("\n=== Mushroom Classification CLI ===")
    print("Enter the following details about the mushroom:")
    
    sample = []
    for col in X.columns:
        value = int(input(f"Enter encoded value for {col}: "))
        sample.append(value)
    
    prediction = model.predict([sample])
    
    if prediction[0] == 0:
        print("\nResult: This mushroom is EDIBLE")
    else:
        print("\nResult: This mushroom is POISONOUS")

predict_mushroom()