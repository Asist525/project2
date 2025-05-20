from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from django.shortcuts import render
from django.http import HttpResponse
import os
from django.conf import settings
@csrf_exempt
def trigger_backtest(request):
    if request.method == "POST":
        data = json.loads(request.body)
        ticker = data.get("ticker")
        start = data.get("start_date")
        end = data.get("end_date")
        return JsonResponse({"status": "성공", "ticker": ticker, "start": start, "end": end})
    return JsonResponse({"error": "POST만 허용됨"}, status=405)

def backtest_page(request):
    path = os.path.join(settings.BASE_DIR, "trading", "templates", "trading", "backtest_page.html")
    if not os.path.exists(path):
        return HttpResponse(f"❌ 템플릿 경로가 존재하지 않음: {path}")
    return render(request, "trading/backtest_page.html")