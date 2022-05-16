from django.urls import path
from ml_model_app import views

urlpatterns = [
     ###Url de la vista predict_condition
     path('predict_condition', views.predict_condition),
     
     ###Url de la vista predict_condition
     path('predict_sentiment', views.predict_sentiment),

]