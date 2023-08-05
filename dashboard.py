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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

st.set_page_config(
    page_title="Online Shoppers Purchasing Intention",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Using Machine Learning to Understand Online Shoppers Purchasing Intention')
st.write('Explore and visualize the dataset')

@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv("online_shoppers_intention.csv")
    return data

data = load_data()

st.subheader('Dataset')
st.dataframe(data)

# Convert 'Month' column to numerical representation using one-hot encoding
data = pd.concat([data, pd.get_dummies(data['Month'], prefix='Month')], axis=1).drop(['Month'], axis=1)

# Convert 'VisitorType' column to numerical representation using one-hot encoding
data = pd.concat([data, pd.get_dummies(data['VisitorType'], prefix='VisitorType')], axis=1).drop(['VisitorType'], axis=1)

def compare_function(option1, option2):
    st.header("Compare Options")
    # Define X and y.
    y = data['Revenue']
    X = data.drop(['Revenue'], axis=1)

    x1 = data
    # removing the target column revenue from x
    x1 = x1.drop(['Revenue'], axis=1)
    y1 = data['Revenue']


    # Tuning
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=101)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=101)

    # Balance
    x1_baseTrain, x1_baseTest, y1_baseTrain, y1_baseTest = train_test_split(x1, y1, test_size=0.3, random_state=101)

    # Load the Tuning models
    naive_bayes_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/Naive_Bayes.sav', 'rb'))
    knn_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/KNN.sav', 'rb'))
    random_forest_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/Random_Forest.sav', 'rb'))
    logistic_regression_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/Logistic_Regression.sav', 'rb'))
    ada_boost_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/ADA_Boosting.sav', 'rb'))
    gardient_boost_model = pickle.load(open('C:/Users/Ee/Desktop/FYP/Gardient_Boost.sav', 'rb'))

    # Load the Balance models
    naive_bayes_model_Balance = pickle.load(open('C:/Users/Ee/Desktop/FYP/Naive_Bayes_Balance.sav', 'rb'))
    knn_model_Balance = pickle.load(open('C:/Users/Ee/Desktop/FYP/KNN_Balance.sav', 'rb'))
    random_forest_model_Balance = pickle.load(open('C:/Users/Ee/Desktop/FYP/Random_Forest_Balance.sav', 'rb'))
    logistic_regression_model_Balance = pickle.load(open('C:/Users/Ee/Desktop/FYP/Logistic_Regression_Balance.sav', 'rb'))
    ada_boost_model_Balance = pickle.load(open('C:/Users/Ee/Desktop/FYP/ADA_Boosting_Balance.sav', 'rb'))
    gardient_boost_model_Balance = pickle.load(open('C:/Users/Ee/Desktop/FYP/Gardient_Boost_Balance.sav', 'rb'))

    # Function to display metrics and ROC curve
    def display_metrics(true_labels, predicted_labels, performance_type, model_name):
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)

        st.subheader(performance_type + " - " + model_name)
        st.write('Accuracy:', accuracy)
        st.write('F1 Score:', f1)
        st.write('Precision:', precision)
        st.write('Recall:', recall)
        st.subheader('Confusion Matrix')
        st.write(cm)

        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - ' + model_name)
        plt.legend(loc="lower right")
        plt.show()
        st.pyplot(plt)

        # Create a dictionary to store the metrics
        metrics_dict = {
            "Metric": ["Accuracy", "F1 Score", "Precision", "Recall"],
            "Value": [accuracy, f1, precision, recall]
        }

        # Convert the dictionary to a DataFrame
        metrics_df = pd.DataFrame(metrics_dict)

        # Display the metrics table
        st.subheader(performance_type + " - " + model_name)
        st.table(metrics_df)

    # Define and train the Tuning models
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

    gardient_boost_model = GradientBoostingClassifier()
    gardient_boost_model.fit(X_train, y_train)

    # Define and train the Balance models
    naive_bayes_model_Balance = GaussianNB()
    naive_bayes_model_Balance.fit(x1_baseTrain, y1_baseTrain)

    knn_model_Balance = KNeighborsClassifier()
    knn_model_Balance.fit(x1_baseTrain, y1_baseTrain)

    random_forest_model_Balance = RandomForestClassifier()
    random_forest_model_Balance.fit(x1_baseTrain, y1_baseTrain)

    logistic_regression_model_Balance = LogisticRegression()
    logistic_regression_model_Balance.fit(x1_baseTrain, y1_baseTrain)

    ada_boost_model_Balance = AdaBoostClassifier()
    ada_boost_model_Balance.fit(x1_baseTrain, y1_baseTrain)

    gardient_boost_model_Balance = GradientBoostingClassifier()
    gardient_boost_model_Balance.fit(x1_baseTrain, y1_baseTrain) 

    # Tuning Model predictions
    if option1 == "Naive Bayes":
        nbm_pred = naive_bayes_model.predict(X_val)
        display_metrics(y_val, nbm_pred, "Tuning Performance", option1)

    elif option1 == "KNN":
        knn_pred = knn_model.predict(X_val)
        display_metrics(y_val, knn_pred, "Tuning Performance", option1)

    elif option1 == "Random Forest":
        rf_pred = random_forest_model.predict(X_val)
        display_metrics(y_val, rf_pred, "Tuning Performance", option1)

    elif option1 == "Logistic Regression":
        lr_pred = logistic_regression_model.predict(X_val)
        display_metrics(y_val, lr_pred, "Tuning Performance", option1)

    elif option1 == "ADA Boost":
        ada_pred = ada_boost_model.predict(X_val)
        display_metrics(y_val, ada_pred, "Tuning Performance", option1)

    elif option1 == "Gradient Boost":
        gbm_pred = gardient_boost_model.predict(X_val)
        display_metrics(y_val, gbm_pred, "Tuning Performance", option1)    


    # Balance Model predictions
    if option2 == "Naive Bayes":
        nbm_balance = naive_bayes_model.predict(x1_baseTest)
        display_metrics(y1_baseTest, nbm_balance, "Balance Performance", option2)

    elif option2 == "KNN":
        knn_balance = knn_model.predict(x1_baseTest)
        display_metrics(y1_baseTest, knn_balance, "Balance Performance", option2)

    elif option2 == "Random Forest":
        rf_balance = random_forest_model_Balance.predict(x1_baseTest)
        display_metrics(y1_baseTest, rf_balance, "Balance Performance", option2)

    elif option2 == "Logistic Regression":
        lr_balance = logistic_regression_model.predict(x1_baseTest)
        display_metrics(y1_baseTest, lr_balance, "Balance Performance", option2)

    elif option2 == "ADA Boost":
        ada_balance = ada_boost_model.predict(x1_baseTest)
        display_metrics(y1_baseTest, ada_balance, "Balance Performance", option2)

    elif option2 == "Gradient Boost":
        gbm_balance = gardient_boost_model_Balance.predict(x1_baseTest)
        display_metrics(y1_baseTest, gbm_balance, "Balance Performance", option2)

    # Perform comparison logic based on the selected options
    st.write("Selected options:")
    st.write("Option 1 (Tuning Model):", option1)
    st.write("Option 2 (Balance Model):", option2)

def main():
    st.title("Compare Function Example")
    st.write("Select options for Tuning Model and Balance Model from the sidebar.")

    # Define options for Tuning Model and Balance Model
    options = ["None", "Naive Bayes", "KNN", "Random Forest", "Logistic Regression", "ADA Boost", "Gradient Boost"]

    # Sidebar layout with custom CSS to place the options side by side
    st.sidebar.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar for the Tuning Model options on the left
    st.sidebar.header("Tuning Model")
    selected_option_1 = st.sidebar.selectbox("Select an option for Tuning", options, key="tuning")

    # Sidebar for the Balance Model options on the right
    st.sidebar.header("Balance Model")
    selected_option_2 = st.sidebar.selectbox("Select an option for Balance", options, key="balance")

    # Only run the comparison if at least one option is selected
    if selected_option_1 != "None" or selected_option_2 != "None":
        compare_function(selected_option_1, selected_option_2)

if __name__ == "__main__":
    main()