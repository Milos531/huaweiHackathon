from django.urls import path
from .views import *


urlpatterns = [
    path('login/', login_req, name='login'),
    path('register/', register, name='register'),
]