from django.db import models

# Create your models here.
class Trade(models.Model):
    Date = models.DateTimeField(db_index=True)
    Ticker = models.CharField(max_length=20, db_index=True)
    Open = models.DecimalField(max_digits=10, decimal_places=6)
    High = models.DecimalField(max_digits=10, decimal_places=6)
    Low = models.DecimalField(max_digits=10, decimal_places=6)
    Close = models.DecimalField(max_digits=10, decimal_places=6)
    Volume = models.BigIntegerField()

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["Date", "Ticker"], name="unique_trade_per_day")
        ]

    def __str__(self):
        return f"{self.Date.strftime('%Y-%m-%d')} - {self.Ticker} (Open: {self.Open}, Close: {self.Close})"
