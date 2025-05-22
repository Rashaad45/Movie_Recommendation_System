from django.urls import path
from . import views 

urlpatterns = [
    path("genai", views.genai, name="genai"),
    path("form", views.form, name="form"),
    path("AI", views.AI, name="AI"),
]