from django.urls import path
from . import views

urlpatterns = [
    path("run-backtest/", views.trigger_backtest, name="run-backtest"),
    path("backtest-page/", views.backtest_page, name="backtest-page"),
]
