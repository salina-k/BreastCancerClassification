from django.http import HttpResponse
from django.shortcuts import render
import joblib


def home(request):
    return render(request, "form.html")


def result(request):
    classifier = joblib.load('model.sav')
    lis = []
    lis.append(request.GET['area'])
    lis.append(request.GET['perimeter'])
    lis.append(request.GET['radius'])
    lis.append(request.GET['concavity'])
    print(lis)
    ans = classifier.predict([lis])
    if ans == 'M':
        ans = 'Malignant'
    else:
        ans = 'Benign'
    return render(request, "result.html", {'ans': ans})
