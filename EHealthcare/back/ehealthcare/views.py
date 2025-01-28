from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.contrib.auth import login, authenticate, logout
from .models import *


def login_req(request):
    if (request.method == "POST"):
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)

        if (user):
            login(request, user)
            token = {
                'username' : user.username,
                'email' : user.email,
                'type' : user.type,
                'surname' : user.surname,
                'forename' : user.forename
            }
            return JsonResponse(user)
    return JsonResponse({"message": "Wrong Request!"})

def register(request):
    if (request.method == "POST"):
        username = request.POST['username']
        old_user = EUser.objects.filter(username=username)
        if old_user:
            return JsonResponse({"message": "Already registered username!"})
        password = request.POST['password']
        email = request.POST['email']
        old_user = EUser.objects.filter(email=email)
        if old_user:
            return JsonResponse({"message": "Already registereed email!"})
        type = request.POST['type']
        surname = request.POST['surname']
        forename = request.POST['forename']
        user = EUser(username = username, password = password, email = email, surname = surname, forename = forename, type = type )
        user.save()
        return JsonResponse({"message": "User registered!"})

    return JsonResponse({"message": "Wrong Request!"})

