import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

st.title('Using Machine Learning to understand Online Shoppers Purchasing Intention')
st.write('Explore and visualize the dataset')

data = pd.read_csv("online_shoppers_intention.csv")

dff = pd.concat([data,pd.get_dummies(data['Month'], prefix='Month')], axis=1).drop(['Month'],axis=1)
dff = pd.concat([dff,pd.get_dummies(dff['VisitorType'], prefix='VisitorType')], axis=1).drop(['VisitorType'],axis=1)

# Define X and y.
y = dff['Revenue']
X = dff.drop(['Revenue'], axis=1)

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=101)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=101)

# # Load the models
# naive_bayes_model = pickle.load(open('C:/Users/USER/Desktop/FYP/Naive_Bayes.sav', 'rb'))
# knn_model = pickle.load(open('Naive_Bayes.sav', 'rb'))
# random_forest_model = pickle.load(open('Naive_Bayes.sav', 'rb'))
# logistic_regression_model = pickle.load(open('Naive_Bayes.sav', 'rb'))
# ada_boost_model = pickle.load(open('Naive_Bayes.sav', 'rb'))


# Sidebar
st.sidebar.header('Model for prediction')
model_name = st.sidebar.selectbox("Please select a model", ("None", "Naive Bayes", "KNN", "Random Forest", "Logistic Regression", "ADA Boost"))

if model_name == "None":
    st.write(dff)

else:
    # Define X and y.
    y = dff['Revenue']
    X = dff.drop(['Revenue'], axis=1)

    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=101)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=101)

    # Load the models
    naive_bayes_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/Naive_Bayes.sav', 'rb'))
    knn_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/KNN.sav', 'rb'))
    random_forest_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/Random_Forest.sav', 'rb'))
    logistic_regression_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/Logistic_Regression.sav', 'rb'))
    ada_boost_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/ADA_Boosting.sav', 'rb'))

    # Function to display metrics
    def display_metrics(true_labels, predicted_labels):
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)

        st.subheader('Model Performance')
        st.write('Accuracy:', accuracy)
        st.write('F1 Score:', f1)
        st.write('Precision:', precision)
        st.write('Recall:', recall)
        st.subheader('Confusion Matrix')
        st.write(cm)

        # Create a heatmap of the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        st.pyplot(plt.gcf())

    # Define and train the models
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train, y_train)

    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)

    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train, y_train)

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train, y_train)

    ada_boost_model = AdaBoostClassifier()
    ada_boost_model.fit(X_train, y_train)

    # Model predictions
    if model_name == "Naive Bayes":
        nbm_pred = naive_bayes_model.predict(X_val)
        display_metrics(y_val, nbm_pred)

    elif model_name == "KNN":
        knn_pred = knn_model.predict(X_val)
        display_metrics(y_val, knn_pred)

    elif model_name == "Random Forest":
        rf_pred = random_forest_model.predict(X_val)
        display_metrics(y_val, rf_pred)

    elif model_name == "Logistic Regression":
        lr_pred = logistic_regression_model.predict(X_val)
        display_metrics(y_val, lr_pred)

    elif model_name == "ADA Boost":
        ada_pred = ada_boost_model.predict(X_val)
        display_metrics(y_val, ada_pred)