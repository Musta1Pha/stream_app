# Lab_12 : Classification des fleurs iris
# Réalisé par KHLIL Mustapha EMSI 2023 - 2024

# Library import (pip install scikit-learn)
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd

# Step 1: DataSet
iris = datasets.load_iris()

# Step 2: Model
models = {
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Bagging': BaggingClassifier()
}

# Model Descriptions
model_descriptions = {
    'Random Forest': 'Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.',
    'K-Nearest Neighbors': 'K-Nearest Neighbors is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation.',
    'Support Vector Machine': 'Support Vector Machine is a supervised machine learning algorithm that can be used for classification or regression tasks. It performs classification by finding the hyperplane that best divides a dataset into classes.',
    'Logistic Regression': 'Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome.',
    'Decision Tree': 'Decision Tree is a flowchart-like structure in which each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label.',
    'Naive Bayes': 'Naive Bayes is a probabilistic algorithm based on the Bayes theorem, which assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.',
    'Neural Network': 'Neural Network is a machine learning model inspired by the structure and functioning of the human brain. It consists of layers of interconnected nodes, or neurons.',
    'AdaBoost': 'AdaBoost is an ensemble learning method that aims to convert a set of weak learners into a strong learner. It assigns more weight to the misclassified data points so that subsequent weak learners focus more on them.',
    'Gradient Boosting': 'Gradient Boosting is an ensemble learning method that builds a series of weak learners and combines them to create a strong learner. It builds trees sequentially, with each tree correcting the errors of the previous ones.',
    'Bagging': 'Bagging (Bootstrap Aggregating) is an ensemble learning method that builds multiple independent models on different subsets of the training data and combines their predictions.',
}

# Sidebar
st.sidebar.header('Iris features')

# Sidebar for Model Selection
selected_model = st.sidebar.selectbox('Select Model', list(models.keys()))

# Step 3: Train
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = models[selected_model]
model.fit(X_train, y_train)

# Step 4: Test and Display Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Model deployment with streamlit: streamlit run lab_12_KHLILMustapha.py (pip install streamlit)
st.header('Iris Classification Model')
st.image('Lab1/images/iris_category.jpeg')

st.write(iris.data)
st.write(iris.target_names)
st.subheader('Model Accuracy')
st.write(f"Accuracy: {accuracy:.2f}")

def user_input():
    sepal_length = st.sidebar.slider('sepal length', 4.3, 7.9, 6.)
    sepal_width = st.sidebar.slider('sepal width', 2.0, 4.4, 3.)
    petal_length = st.sidebar.slider('petal length', 1., 9.2, 2.)
    petal_width = st.sidebar.slider('petal width', 0.5, 2.5, 1.2)

    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }

    flower_features = pd.DataFrame(data, index=[0])
    return flower_features

df = user_input()
st.write(df)
st.subheader('Prediction')

prediction = model.predict(df)
st.write(iris.target_names[prediction][0])
st.image("Lab1/images/" + iris.target_names[prediction][0] + ".png")

# Display Model Description at the bottom
st.sidebar.subheader('Model Description')
st.sidebar.write(model_descriptions[selected_model])


# Améliorations :
#DONE 1 - Avec une commande , Au niveau de prediction on affiche value + image
#DONE 2 - Ajouter des algorithmes +10
#DONE 3 - Effectuer choix de l'algorithme souhaité (streamlit selection gpt)
#DONE 4 - Expliquer le model
#DONE 5 - accuracy


