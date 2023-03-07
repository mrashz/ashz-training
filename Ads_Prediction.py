import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.write("""
# Predict number of sales based on the allocation of advertisement channel.
""")

st.sidebar.header('User Input Parameters')

#'TV', 'Radio', 'Newspaper'

def user_input_features():
    ads_TV = st.sidebar.slider('Advertisement on TV', 0, 500, 500)
    ads_Radio = st.sidebar.slider('Advertisement on Radio', 0, 500, 500)
    ads_Newspaper = st.sidebar.slider('Advertisement on Newspaper', 0, 500, 500)
    data = {'ads_TV': ads_TV,
            'ads_Radio': ads_Radio,
            'ads_Newspaper': ads_Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('User Input parameters')
st.write(df_input)

# Load dataset into a pandas dataframe
df_train = pd.read_csv('Advertising.csv')

#st.subheader('View Data')
#st.write(df_train)

# Define features and target variable
X = df_train[['TV', 'Radio', 'Newspaper']]
y = df_train['Sales']

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X, y)

# Predict the number of sales for new data
predicted_sales = model.predict(df_input)


st.subheader('Prediction')
st.write(predicted_sales)

