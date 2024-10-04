from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('search-area/', views.search_area, name='search_area'),
    path('search-result/', views.search_result, name='search_result'),
    path('predict-year/', views.predict_year, name='predict_year'),
    path('predict-result/', views.predict_result, name='predict_result'),
    path('chat/', views.chat, name='chat'),
]
