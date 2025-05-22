from django.shortcuts import render
from .models import Moviebase
from .models import upcoming

# Create your views here.
def index(request):

    movies = Moviebase.objects.all()
    
    return render(request,'index.html',{'movies':movies})


def soon(request):

    coming = upcoming.objects.all()

    return render(request,'index.html',{'coming':coming})

