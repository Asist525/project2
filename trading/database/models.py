from django.db import models
from django.utils import timezone

class Stock(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)
    market = models.CharField(max_length=20)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.symbol} - {self.name}"
