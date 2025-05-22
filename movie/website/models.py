from django.db import models

# Create your models here.
class Moviebase(models.Model):
    rating    =  models.IntegerField()
    name      =  models.CharField(max_length=100)
    duration  =  models.CharField(max_length=100)
    genre     =  models.CharField(max_length=100)
    certificate= models.CharField(max_length=100)
    img       =  models.ImageField(upload_to='pics')
    link      =  models.CharField(max_length=100)

class upcoming(models.Model):
    rating    =  models.IntegerField()
    name      =  models.CharField(max_length=100)
    duration  =  models.CharField(max_length=100)
    genre     =  models.CharField(max_length=100)
    certificate= models.CharField(max_length=100)
    img       =  models.ImageField(upload_to='pics')

