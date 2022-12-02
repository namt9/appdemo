from django.db import models

# Create your models here.


class ImageModel(models.Model):
    image = models.ImageField(null=True, blank=True , upload_to="img")

