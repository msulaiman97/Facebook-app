import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
#Loadind Data 
st.set_page_config(layout='wide')
df=pd.read_csv("facebook_ads.csv", encoding="ISO-8859-1")
option=st.sidebar.selectbox("Pick a Choise:",['Home','EDA','Model'])
if option=='Home':
    st.title("Facbook App")
    st.text("Author:Mohamed Sulaiman")
    st.dataframe(df.head())       
elif option=='EDA':
    st.title("Facbook App EDA")
    col1,col2=st.columns(2)
    fig=px.scatter(data_frame=df,x='Time Spent on Site',y='Salary') 
    st.plotly_chart(fig) 
    with col1:
        fig=px.violin(data_frame=df,x='Time Spent on Site')
        st.plotly_chart(fig)    
    fig=plt.figure()
    df['Clicked'].value_counts().plot(kind='bar')
    st.pyplot(fig)
elif option=='Model':
    st.title("Ads Clicked Prediction")
    st.text("In this App, We will Predict the ads Click using Salary and Time Spent")
    st.text("Please Enter The Following Values:")
    #Buliding Model
    time=st.number_input("Enter time spend on Website")
    salary=st.number_input("Enter Salary")
    btn=st.button("Submit")
    df.drop(columns=['Names','emails','Country'],inplace=True)
    X=df.drop("Clicked",axis=1)
    y=df['Clicked']
    ms=MinMaxScaler()
    X=ms.fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
    clf=LogisticRegression()
    clf=pickle.load(open('My_model.pkl','rb'))
    result=clf.predict(ms.fit_transform([[time,salary]]))
    if btn:
        if result==1:
            st.wrtie("Clicked")
        elif result==0:
            st.write("Not Clicked")
