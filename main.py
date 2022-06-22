import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

#load the dataset to pandas dataframe
URL = "http://bit.ly/w-data"
student_data = pd.read_csv(URL)

#Prepare data
X = student_data.copy()
y = student_data['Scores']
del X['Scores']

#create a machine learning model and train it
lineareg = LinearRegression()
lineareg.fit(X,y)
print('Accuracy score : ',lineareg.score(X,y),'\n')


#function to predict the input hours
def predict_score(hours):
    hours = np.array(hours) #process input
    pred_score = lineareg.predict(hours.reshape(-1,1)) #prediction
    return np.round(pred_score[0], 2)


input = gr.inputs.Number(label='Number of Hours studied')
output = gr.outputs.Textbox(label='Predicted Score')

gr.Interface(
    predict_score,
    input,
    output,
    live=True
).launch()
