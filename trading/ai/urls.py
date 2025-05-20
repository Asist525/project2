# ai/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("short-reward-chart/", views.short_reward_chart_page, name="short_reward_chart"),
    path("long-reward-chart/", views.long_reward_chart_page, name="long_reward_chart"),
    path("reward_chart_data/", views.reward_chart_view, name="reward_chart_data"),  # Short-term
    path("reward_chart_data2/", views.reward_chart_view2, name="reward_chart_data2"),  # Long-term
]
