from django.db import models

# Create your models here.

class AI_REWARD(models.Model):
    # 성능 지표 (소수점 6자리까지 저장, 필요한 경우 더 늘릴 수 있음)
    MAPE = models.DecimalField(max_digits=10, decimal_places=6, verbose_name="Mean Absolute Percentage Error")
    MAE = models.DecimalField(max_digits=10, decimal_places=6, verbose_name="Mean Absolute Error")
    RMSE = models.DecimalField(max_digits=10, decimal_places=6, verbose_name="Root Mean Square Error")
    R2 = models.DecimalField(max_digits=10, decimal_places=6, verbose_name="R-Squared")
    REWARD = models.DecimalField(max_digits=10, decimal_places=6, verbose_name="Reward")
    
    # 생성 및 수정 시간 자동 기록
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Updated At")

    # 선택적으로 추가할 필드
    model_name = models.CharField(max_length=100, blank=True, null=True, verbose_name="Model Name")
    notes = models.TextField(blank=True, null=True, verbose_name="Additional Notes")

    def __str__(self):
        return f"Reward Record ({self.id}): {self.REWARD}"
    
class AI_REWARD2(models.Model):
    MCC = models.FloatField(default=0.0)
    SMAPE = models.FloatField(default=0.0)
    REWARD = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"MCC: {self.MCC:.4f}, SMAPE: {self.SMAPE:.4f}, REWARD: {self.REWARD:.4f}"