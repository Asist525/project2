from django.urls import path
from . import views

urlpatterns = [
    path("", views.stock_list, name="stock_list"),  # 기본 경로
]
