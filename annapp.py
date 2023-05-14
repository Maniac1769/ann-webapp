import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models

pickle_in=open('ann.pkl','rb')
ann_model = pickle.load(pickle_in)



dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



st.title('Churn Modellin using AI')
    
    
    # getting the input data from the user
col1, col2, col3 = st.columns(3)
    
with col1:
    geographyonehot_1 = st.text_input('geo code')
        
with col2:
    geographyonehot_2 = st.text_input('geo code')
    
with col3:
    geographyonehot_3 = st.text_input('geo code')
    
with col1:
    creditscore = st.text_input('credit score')
    
with col2:
    gender = st.text_input('gender')
    
with col3:
    age = st.text_input('age')
    
with col1:
    tenure= st.text_input('tenure')
    
with col2:
    balance= st.text_input('balance')

with col3:
    numberofproducts=st.text_input('number of products')
with col1:
    hascrcard=st.text_input('has credit card?')
with col2:
    isactivemember=st.text_input('is active member?')
with col3:
    estimatedsalary=st.text_input('estimated salary')

churn_diagnosis = ''
    
    # creating a button for Prediction
    
if st.button('Churn modelling Result'):
    churn_prediction = ann_model.predict(sc.transform([[geographyonehot_1,geographyonehot_2,geographyonehot_3,creditscore,gender,age,tenure,balance,numberofproducts,hascrcard,isactivemember,estimatedsalary]]))
        
    if (churn_prediction[0] == 1):
        churn_diagnosis = 'The customer stays'
    else:
        churn_diagnosis = 'The customer wont stay'
        
st.success(churn_diagnosis)
