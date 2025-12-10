import pandas as pd
import matplotlib.pyplot as plt
import skops.io as sio

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Load and shuffle dataset
drug_df = pd.read_csv("data/drug200.csv")
drug_df = drug_df.sample(frac=1)

# Train-test split
X = drug_df.drop("Drug", axis=1).values
y = drug_df["Drug"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

# Define preprocessing and model pipeline
cat_col = [1, 2, 3]
num_col = [0, 4]

transform = ColumnTransformer([
    ("encoder", OrdinalEncoder(), cat_col),
    ("num_imputer", SimpleImputer(strategy="median"), num_col),
    ("num_scaler", StandardScaler(), num_col),
])

pipe = Pipeline(steps=[
    ("preprocessing", transform),
    ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
])

# Train the model
pipe.fit(X_train, y_train)

# Make predictions and evaluate
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

# Save metrics
with open("results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}")

# Save confusion matrix
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("results/model_results.png", dpi=120)

# Save model
sio.dump(pipe, "model/drug_pipeline.skops")