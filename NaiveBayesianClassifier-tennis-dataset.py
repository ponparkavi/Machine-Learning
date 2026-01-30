import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def load_data(file_path):
  df = pd.read_csv(file_path)
  return df
  
def preprocess_data(df):
  label_encoders = {}
  for column in df.columns[:-1]:
    if df[column].dtype == 'object':
      le = LabelEncoder()
      df[column] = le.fit_transform(df[column])
      label_encoders[column] = le
    target_le = LabelEncoder()
    df[df.columns[-1]] = target_le.fit_transform(df[df.columns[-1]])
    label_encoders[df.columns[-1]] = target_le
  return df, label_encoders
  
def train_naive_bayes(X_train, y_train):
  model = GaussianNB()
  model.fit(X_train,y_train)
  return model
  
def compute_accuracy(model, X_test, y_test):
  y_pred = model.predict(X_test)
  return accuracy_score(y_test, y_pred), y_pred
  
if __name__ == "__main__":
  file_path = "tennis.csv"
  df = load_data(file_path)
  df, encoders = preprocess_data(df)
  
  X = df.iloc[:, :-1].values
  y = df.iloc[:, -1].values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  print("\nThe First 5 values of train data (original):\n")
  print(pd.read_csv(file_path).iloc[:5, :-1])
  
  print("\nThe First 5 values of Train output (original):\n")
  print(pd.read_csv(file_path).iloc[:5, -1])
  
  print("\nNow the Train data (encoded):\n")
  print(pd.DataFrame(X_train[:5], columns=df.columns[:-1]))
  print("\nNow the Train output (encoded):\n", y_train[:5])
  model = train_naive_bayes(X_train, y_train)
  accuracy, _ = compute_accuracy(model, X_test, y_test)
  print(f"\nAccuracy is:{accuracy}\n")
