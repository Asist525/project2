# trading/urls.py
from django.urls import path
from django.conf.urls.static import static
from . import views
from django.conf import settings  

urlpatterns = [
    path("backtest-page/", views.backtest_page, name="backtest-page"),
    path("run-backtest/", views.trigger_backtest, name="run-backtest"),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
