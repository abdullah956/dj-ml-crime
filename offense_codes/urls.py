from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('search-area/', views.search_area, name='search_area'),
    path('predict-year/', views.predict_year, name='predict_year'),
    path('chat/', views.chat, name='chat'),
]
