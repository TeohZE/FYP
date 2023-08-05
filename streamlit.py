import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler


data = pd.read_csv("online_shoppers_intention.csv")

dff = pd.concat([data, pd.get_dummies(data['Month'], prefix='Month')], axis=1).drop(['Month'], axis=1)
dff = pd.concat([dff, pd.get_dummies(dff['VisitorType'], prefix='VisitorType')], axis=1).drop(['VisitorType'], axis=1)

dff = pd.concat([data, pd.get_dummies(data['Month'], prefix='Month')], axis=1).drop(['Month'], axis=1)
dff = pd.concat([dff, pd.get_dummies(dff['VisitorType'], prefix='VisitorType')], axis=1).drop(['VisitorType'], axis=1)

y = dff['Revenue']
X = dff.drop(['Revenue'], axis=1)

x1 = dff

x1 = x1.drop(['Revenue'], axis = 1)
y1 = data['Revenue']

#Tuning
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=101)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=101)

#Balance
x1_baseTrain, x1_baseTest, y1_baseTrain, y1_baseTest = train_test_split(x1, y1, test_size = 0.3, random_state = 101)

classifiers = {
    "None": None,
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "ADA Boost": AdaBoostClassifier()
}

def perform_tuning(X_train, X_test, y_train, y_test, classifier):
    # Perform tuning here
    # Replace this with your own tuning implementation

    tuned_classifier = classifier  # Placeholder for tuned classifier
    tuned_predictions = tuned_classifier.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, tuned_predictions)
    f1 = f1_score(y_test, tuned_predictions)
    precision = precision_score(y_test, tuned_predictions)
    recall = recall_score(y_test, tuned_predictions)
    cm = confusion_matrix(y_test, tuned_predictions)
    
    return accuracy, f1, precision, recall, cm

def perform_balancing(X_train, X_test, y_train, y_test, classifier):
    # Perform balancing here using RandomOverSampler
    oversampler = RandomOverSampler()
    X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)
    
    classifier.fit(X_train_balanced, y_train_balanced)
    balanced_predictions = classifier.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, balanced_predictions)
    f1 = f1_score(y_test, balanced_predictions)
    precision = precision_score(y_test, balanced_predictions)
    recall = recall_score(y_test, balanced_predictions)
    cm = confusion_matrix(y_test, balanced_predictions)
    
    return accuracy, f1, precision, recall, cm

# Streamlit app
st.title("Classifier Comparison")
classifier_choice = st.selectbox("Select Classifier", list(classifiers.keys()))

# Load and preprocess your data here
# Replace this with your own data loading and preprocessing
knn_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/KNN.sav', 'rb'))
random_forest_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/Random_Forest.sav', 'rb'))
logistic_regression_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/Logistic_Regression.sav', 'rb'))
ada_boost_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/ADA_Boosting.sav', 'rb'))

# Load the Balance models
naive_bayes_model_Balance = pickle.load(open('C:/Users/Ee/Desktop/FYP/Naive_Bayes_Balance.sav', 'rb'))
knn_model_Balance = pickle.load(open('C:/Users/Ee/Desktop/FYP/KNN_Balance.sav', 'rb'))
random_forest_model_Balance = pickle.load(open('C:/Users/Ee/Desktop/FYP/Random_Forest_Balance.sav', 'rb'))
logistic_regression_model_Balance = pickle.load(open('C:/Users/Ee/Desktop/FYP/Logistic_Regression_Balance.sav', 'rb'))
ada_boost_model_Balance = pickle.load(open('C:/Users/Ee/Desktop/FYP/ADA_Boosting_Balance.sav', 'rb'))



#Tuning
dff = pd.concat([data, pd.get_dummies(data['Month'], prefix='Month')], axis=1).drop(['Month'], axis=1)
dff = pd.concat([dff, pd.get_dummies(dff['VisitorType'], prefix='VisitorType')], axis=1).drop(['VisitorType'], axis=1)



# Perform tuning and balance method
if classifier_choice != "None":
    classifier = classifiers[classifier_choice]
    
    if classifier_choice == "Naive Bayes":
        # Naive Bayes doesn't require tuning, skip to balancing
        accuracy_tuning, f1_tuning, precision_tuning, recall_tuning, cm_tuning = None, None, None, None, None
    else:
        accuracy_tuning, f1_tuning, precision_tuning, recall_tuning, cm_tuning = perform_tuning(X_train, X_test, y_train, y_test, classifier)

    accuracy_balancing, f1_balancing, precision_balancing, recall_balancing, cm_balancing = perform_balancing(X_train, X_test, y_train, y_test, classifier)
    
    
    st.subheader("Tuning Method")
    st.write("Accuracy:", accuracy_tuning)
    st.write("F1 Score:", f1_tuning)
    st.write("Precision:", precision_tuning)
    st.write("Recall:", recall_tuning)
    st.subheader("Confusion Matrix (Tuning Method)")
    st.write(cm_tuning)
    
    st.subheader("Balance Method")
    st.write("Accuracy:", accuracy_balancing)
    st.write("F1 Score:", f1_balancing)
    st.write("Precision:", precision_balancing)
    st.write("Recall:", recall_balancing)
    st.subheader("Confusion Matrix (Balance Method)")
    st.write(cm_balancing)
else:
    st.warning("Please select a valid classifier.")

# # Run the Streamlit app
# if __name__ == "__main__":
#     main()