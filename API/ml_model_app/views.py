from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .serializers import ReviewSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from condition_model import *


################# view para predecir la condicion ###################
@api_view(["POST"])

def predict_condition(request):
    
    serializer =  ReviewSerializer(data =  request.data)

    if serializer.is_valid():
        serializer.save()

    predict = predictcondition(serializer)
   
    return Response(predict)




################### view para predecir el sentimiento #####################

from sentiment_model import *
@api_view(["POST"])

def predict_sentiment(request):
    
    serializer =  ReviewSerializer(data =  request.data)

    if serializer.is_valid():
        serializer.save()

    review = serializertoarray(serializer)
    predict = sentiment_prediction(review)
   
    return Response(predict)