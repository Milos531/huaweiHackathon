from django.db import models
from django.contrib.auth.models import User, AbstractUser


class EUser(AbstractUser):
    type = models.CharField(max_length=50, null=False)
