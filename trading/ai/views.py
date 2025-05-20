# ai/views.py
from collections import defaultdict
from django.http import JsonResponse
from django.utils import timezone
from datetime import timedelta
from math import floor
from ai.models import AI_REWARD, AI_REWARD2   # short / long
from django.shortcuts import render

# --- 페이지 ----------------------------------------------------------------
def short_reward_chart_page(request):
    return render(request, "ai/short_reward_chart.html")

def long_reward_chart_page(request):
    return render(request, "ai/long_reward_chart.html")

# --- 데이터 API -------------------------------------------------------------
def reward_chart_view(request):          # Short-term (5개 지표)
    return _generate_reward_data(
        model=AI_REWARD,
        request=request,
        fields=("MAPE", "MAE", "RMSE", "R2", "REWARD"),
    )

def reward_chart_view2(request):         # Long-term (3개 지표)
    return _generate_reward_data(
        model=AI_REWARD2,
        request=request,
        fields=("MCC", "SMAPE", "REWARD"),
    )

# --- 공통 유틸 --------------------------------------------------------------
def _generate_reward_data(model, request, fields):
    interval   = int(request.GET.get("interval", 10))      # 분
    days_back  = int(request.GET.get("days", 7))           # 기본 7일
    since      = timezone.now() - timedelta(days=days_back)

    qs = (
        model.objects
             .filter(created_at__gte=since)
             .values("created_at", *fields)
             .order_by("created_at")
    )

    buckets = defaultdict(lambda: {f: [] for f in fields})
    interval_sec = interval * 60

    for row in qs:
        ts = row["created_at"].timestamp()
        bucket = floor(ts / interval_sec) * interval_sec
        for f in fields:
            buckets[bucket][f].append(float(row[f]))

    # ─── JSON 직렬화 ────────────────────────────────────────────────────────
    labels   = []
    datasets = {f: [] for f in fields}

    for bk in sorted(buckets):
        labels.append(timezone.datetime.fromtimestamp(bk).strftime("%Y-%m-%d %H:%M"))
        for f in fields:
            vals = buckets[bk][f]
            datasets[f].append(sum(vals) / len(vals))

    data = {
        "labels": labels,
        "datasets": [
            {"label": f, "data": datasets[f]} for f in fields
        ],
    }
    resp = JsonResponse(data)
    resp["Cache-Control"] = "no-store"
    return resp
