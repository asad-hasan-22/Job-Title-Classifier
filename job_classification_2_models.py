import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle

# Load vectorizor
loaded_vectorizer = pickle.load(open('count_vectorizer.pkl', 'rb'))

# Function that converts to standardized job title
@st.cache
def raw_job_title_to_onet(clf, job_title):
    print(clf.predict(loaded_vectorizer.transform([job_title])))
    standardized_job_title = clf.predict(loaded_vectorizer.transform([job_title]))
    return standardized_job_title

# Streamlit formatting
st.title('Job Title Classifier')
st.subheader('Converts a job title to a standardized version')

# Model Selector
option = st.selectbox('Select a model: ', ('Naive Bayes', 'Support Vector Machine (SVM)'))
#filename = 'naive_bayes_1.sav'

# Condition on model selection
if option == 'Naive Bayes':
    filename = '20211201_naive_bayes_job_title_classifier.pkl'
elif option == 'Support Vector Machine (SVM)':
    filename = '20211201_linear_svc_job_title_classifier.pkl'

# Load model
loaded_model = pickle.load(open(filename, 'rb'))

job_title = st.text_input("Enter a job title: ")
standardized_job_title = raw_job_title_to_onet(loaded_model, job_title)
st.write('Standardized job title:')
st.success(standardized_job_title[0])
#st.balloons()
