from importlib.metadata import files
from pyexpat import model
from rest_framework import serializers
from django.db import models
from django.db.models import fields
#importamos el modelo review
from .models import Review
#creacion del zerializer.
class ReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = Review
        fields = '__all__'