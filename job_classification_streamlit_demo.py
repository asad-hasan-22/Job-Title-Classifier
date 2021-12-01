import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
#from naive_bayes_streamlit import raw_job_title_to_onet

print("sldkjfa;lsdkfj")
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
#count_vect = CountVectorizer()

# function to predict from text
@st.cache
def raw_job_title_to_onet(clf, job_title):
    print(clf.predict(loaded_vectorizer.transform([job_title])))
    standardized_job_title = clf.predict(loaded_vectorizer.transform([job_title]))
    return standardized_job_title

filename = 'naive_bayes_1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#top_x = st.slider("lskadjf;alsdjkf")
st.title('Job Title Classifier')
st.subheader('Converts a job title to a standardized version')

option = st.selectbox('Select a model: ', ('Naive Bayes', 'Support Vector Machine (SVM)'))


job_title = st.text_input("Enter a job title: ")
standardized_job_title = raw_job_title_to_onet(loaded_model, job_title)
#st.write("Standardized job title: ", standardized_job_title[0])

st.write('Standardized job title:')
st.success(standardized_job_title[0])
#st.balloons()
