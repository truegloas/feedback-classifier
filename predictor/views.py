from django.shortcuts import render

import os
from tensorflow import saved_model


def home(request):
    return render(request, 'home.html')


def to_correct_predict(raw_predict):
    listed_predict = raw_predict.numpy().tolist()[0]

    correct_predict = listed_predict.index(max(listed_predict))

    return correct_predict


def predict(text):
    loaded_model = saved_model.load('electra_small')
    results = to_correct_predict(loaded_model(text))

    return results


def result(request):
    feedback_text = request.GET['feedback_text']

    prediction = predict(feedback_text)

    return render(request, 'result.html', {'result': prediction})
