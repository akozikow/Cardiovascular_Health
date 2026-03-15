import streamlit as st
import pandas as pd
import xgboost as xgb
import requests
import json

st.set_page_config(layout="wide")
@st.cache_resource
def load_model():
    url = 'https://github.com/akozikow/Cardiovascular_Health/raw/refs/heads/main/xgb_cardio_model.json'
    response = requests.get(url)
    params_json = response.json()
    with open('memory_xgb.json', 'w') as file:
        json.dump(params_json, file)
    loaded_model_memory = xgb.XGBClassifier()
    loaded_model_memory.load_model('memory_xgb.json')
    return loaded_model_memory

model = load_model()

entry_df = pd.DataFrame({
    "age": [0],
    'restingBP' : [0],
    'serumcholestrol' : [0],
    'maxheartrate' : [0],
    'oldpeak' : [0],
    'noofmajorvessels' : [0],
    'gender_0' : [0],
    'gender_1' : [0],
    'chestpain_0': [0],
    'chestpain_1' : [0],
    'chestpain_2' : [0],
    'chestpain_3' : [0],
    'fastingbloodsugar_0' : [0],
    'fastingbloodsugar_1' : [0],
    'restingrelectro_0' : [0],
    'restingrelectro_1' : [0],
    'restingrelectro_2' : [0],
    'exerciseangia_0' : [0],
    'exerciseangia_1' : [0],
    'slope_0' : [0],
    'slope_1' : [0],
    'slope_2' : [0],
    'slope_3' : [0],
})

st.header('Heart Disease Prediction Model')
st.divider()
st.subheader('Background')
st.write('Hello, and welcome to the companion application to my analysis of Cardiovascular Health data. '
        'The data the model was trained on was collected from 1000 patients in India, and can be found at the following link: https://data.mendeley.com/datasets/dzz48mvjht/1. '
         )
st.write('The original analysis was done by Bhanu Prakash Doppala, Debnath Bhattacharyya, Midhunchakkaravarthy Janarthanan, and Namkyun Baik, who put together the paper linked here: https://onlinelibrary.wiley.com/doi/10.1155/2022/2585235. '
         )
st.write('The data used in this analysis was one of three they investigated; the researchers had used an ensemble model, combining multiple ML algorithms together to create a consensus answer. '
         'For my analysis, I found an XGBoost model to be the best classifier. The Jupyter Notebook is shared on my Github. '
         'The model was ~99% accurate on the test set after evaluation on the training and validation sets. '
         'https://github.com/akozikow/Cardiovascular_Health/blob/main/Cardiovascular_Health_Analysis.ipynb. ')
st.write('To make a prediction, simply input patient attributes on the sidebar to the left and a prediction with the confidence level will be displayed below.')
st.divider()

st.sidebar.subheader('Select Patient Attributes')
age = st.sidebar.number_input('Age', min_value = 0, max_value = 120, step = 1, key = 'age')
entry_df.at[0, 'age'] = age
bp = st.sidebar.number_input('Resting Systolic BP', min_value = 0, max_value = 300, step = 1, key = 'bp')
entry_df.at[0, 'restingBP'] = bp
chol = st.sidebar.number_input('Serum Choleseterol', min_value = 0, max_value = 1000, step = 1, key = 'chol')
entry_df.at[0, 'serumcholestrol'] = chol
hr = st.sidebar.number_input('Max Heart Rate', min_value = 0, max_value = 300, step = 1, key = 'hr')
entry_df.at[0, 'maxheartrate'] = hr
oldpeak = st.sidebar.number_input('Oldpeak', min_value = 0.0, max_value = 20.0, step = 0.1, key = 'oldpeak')
entry_df.at[0, 'oldpeak'] = oldpeak
noofmajorvessels = st.sidebar.number_input('Number of Major Vessels', min_value = 0, max_value = 4, step = 1, key = 'vessels')
entry_df.at[0, 'noofmajorvessels'] = noofmajorvessels
gender = st.sidebar.selectbox('Gender', ('Male', 'Female'), key = 'gender')
if gender == 'Female':
    entry_df.at[0, 'gender_0'] = 1
else:
    entry_df.at[0, 'gender_1'] = 1
chestpain = st.sidebar.selectbox('Chest Pain', ('Typical Angina', 'Atypical Angina', 'Non-Angina Pain', 'Asymptomatic'), key = 'chestpain')
if chestpain == 'Typical Angina':
    entry_df.at[0, 'chestpain_0'] = 1
elif chestpain == 'Atypical Angina':
    entry_df.at[0, 'chestpain_1'] = 1
elif chestpain == 'Non-Angina Pain':
    entry_df.at[0, 'chestpain_2'] = 1
elif chestpain == 'Asymptomatic':
    entry_df.at[0, 'chestpain_3'] = 1
bloodsugar = st.sidebar.selectbox('Fasting Blood Sugar', ('Less than 120 mg/dl', 'Greater than 120 mg/dl'), key = 'bloodsugar')
if bloodsugar == 'Less than 120 mg/dl':
    entry_df.at[0, 'fastingbloodsugar_0'] = 1
elif bloodsugar == 'Greater than 120 mg/dl':
    entry_df.at[0, 'fastingbloodsugar_1'] = 1
electro = st.sidebar.selectbox('Resting ECG Results', ('Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'), key = 'restingECG')
if electro == 'Normal':
    entry_df.at[0, 'restingrelectro_0'] = 1
elif electro == 'ST-T Wave Abnormality':
    entry_df.at[0, 'restingrelectro_1'] = 1
elif electro == 'Left Ventricular Hypertrophy':
    entry_df.at[0, 'restingrelectro_2'] = 1
exerciseangina = st.sidebar.selectbox('Exercise Induced Angina', ('No', 'Yes'), key = 'exerciseangina')
if exerciseangina == 'No':
    entry_df.at[0, 'exerciseangia_0'] = 1
elif exerciseangina == 'Yes':
    entry_df.at[0, 'exerciseangia_1'] = 1
slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', ('Typical', 'Upsloping', 'Flat', 'Downsloping'), key = 'slope')
if slope == 'Typical':
    entry_df.at[0, 'slope_0'] = 1
elif slope == 'Upsloping':
    entry_df.at[0, 'slope_1'] = 1
elif slope == 'Flat':
    entry_df.at[0, 'slope_2'] = 1
elif slope == 'Downsloping':
    entry_df.at[0, 'slope_3'] = 1

prediction = model.predict(entry_df)
healthy_prob = (model.predict_proba(entry_df)[0,0])*100
diseased_prob = (model.predict_proba(entry_df)[0,1])*100

predict_button = st.sidebar.button('Predict')

col1, col2 = st.columns(2)
with col1:
    st.subheader('Make a Prediction')
    if prediction == 0 and predict_button == True:
        st.write(f'Model predicts No Heart Disease with {healthy_prob:.2f}% Confidence')
    elif prediction == 1 and predict_button == True:
        st.write(f'Model predicts Heart Disease with {diseased_prob:.2f}% Confidence')
    else:
        st.write('Please enter patient features and press "Predict"')
with col2:
    st.image('https://raw.githubusercontent.com/akozikow/Cardiovascular_Health/main/Cardio Classification Report.jpg', caption='Accuracy of model with blinded testing set.')
st.divider()
st.subheader('Explanation of Model')

st.write('The intention of this model is to assist doctors in evaluating the risk that their patients will develop heart disease of some kind. '
         'Therefore, rather than explaining the terminology of the patient attributes for the layman, I will discuss here the findings of the model. ')
st.write('Tree based XGBoost models are complicated in construction, and do not have "coefficients" that can be simply visualized like linear models, which would indicate which variables are most impactful on the output. '
         'Instead, we will try to understand which variables the model might see as import by looking at the correlation matrix of the features. ')
st.image('https://raw.githubusercontent.com/akozikow/Cardiovascular_Health/main/cardio_correlation_matrix.jpg', caption='Correlation Matrix')
st.write('In the correlation matrix, values closer to 1 or -1 indicate a stronger positive or negative correlation. '
         'Correlation indicates the strength of the linear relationship between two features; in a way, it indicates how predictive one variable is of another. '
         'We will focus on the "target" column of the above correlation matrix. ')
col1, col2 = st.columns(2)
with col1:
    st.write(
        'The graph to the right depicts the correlations between each individual variable in the dataset and the target variable (i.e., whether the patient has heart disease or not). '
        'You will notice many of the variables are appended with an underscore and an integer: this is due to a process called One Hot Encoding which puts the data in a form that is more accessible to the model, which is important to maximize its predictive power. '
        'We will focus on which variables tend to appear at the extreme ends of the barplot.'
    )
    st.write(
        'For example, we see "slope" can be seen at both extreme ends: slope_2 and slope_3 appear at the left, and slope_1 and slope 0 appear at the right. '
        'This indicates that the former have high positive correlation with heart disease, and the latter have high negative correlation. '
        'A variable like "age" appears in the middle, which means it has very little correlation (positive or negative) with heart disease classification, and therefore is likely not very predictive or important to a model. '

    )
with col2:
    st.image('https://raw.githubusercontent.com/akozikow/Cardiovascular_Health/main/correlation_with_target.jpg',
             caption='Variable correlation with heart disease status.')

st.write('XGBoost models use tree structures to make predictions, which split the data via binary true/false classifications. For example, the first split of a tree might be based on whether or not the patient has blood pressure higher than 160. '
         'This splits the data into two (not necessarily equal) parts based on that criteria. This process, called recursive partitioning, builds the tree structure over many such splits. '
         'The intention is that after this process, the tree will be able to split data into homogenous groups where all the data points are similar/the same in terms of the target variable. '
         'XGBoost is powerful because it is what is called an ensemble model; it makes many decision trees, accounts for the difference between their predictions and the truth, and adds their answers together to get a final answer. '
         'The contributions of each "learner" (individual trees) are multiplied by a "learning rate", which dictates how much of that contribution will be added to the final answer. ')
st.divider()
st.header('Interpreting the Results')

st.write('When we evaluate the results of the model, we have to take into account the data that was used to build it. '
         'This data was from India, so all the participants in the study are likely more similar to each other than they would be to participants from another study based in another country. '
         'We should be careful taking a model trained on one population and extrapolating it to another.')
st.write('Additionally, when we see the correlations of our variables with the occurrence of heart disease, we see that the slope of the peak exercise ST Segment is very heavily correlated. '
         'The model learned that the easiest way to split the patients into homogenous healthy/diseased groups was to split along this varaible. '
         'You can try this yourself: put in any values you like for all the other variables, and then just change the "slope" variable and see how the prediction changes. '
         'You will find that just changing the "slope" variable and leaving everything else unchanged will swing the prediction from Healthy to Diseased in many cases. ')
st.write('One must wonder why this is. A quick search will tell you this ST Slope is a key indicator of myocardial ischemia, and while ischemia may be among the most common types, it is not the only type of heart disease. '
         'Did the study originators only collect data for heart disease patients with this particular ailment? This would certainly skew the data. '
         'This highlights the critical need to understand not only the data itself and its summary statistics, etc., but to understand under what conditions the data was collected.')
st.write('This project was chiefly for me to learn more about saving, loading, and deploying predictive models, but it also demonstrated the need to have the history of the dataset you are working with. '
         'The more you know about the who, the what, and the why of the data, the better analysis you will be able to provide to the stakeholders involved. ')