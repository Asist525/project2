from django.shortcuts import render
from django.http import JsonResponse
from .models import Stock

# Create your views here.

def stock_list(request):
    stocks = list(Stock.objects.values())
    return JsonResponse(stocks, safe=False)