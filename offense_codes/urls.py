from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('search-area/', views.search_area, name='search_area'),
    path('search-result/', views.search_result, name='search_result'),
    path('predict-year/', views.predict_year, name='predict_year'),
    path('predict-result/', views.predict_result, name='predict_result'),
    path('chat/', views.chat, name='chat'),
    path('chatbot-response/', views.chatbot_response, name='chatbot_response'),
    path('area_crime_heatmap/', views.area_crime_heatmap, name='area_crime_heatmap'),
    path('predicted_crime_by_area/', views.predicted_crime_by_area_view, name='predicted_crime_by_area'),
    path('crime_rate_by_area/', views.crime_rate_by_area_view, name='crime_rate_by_area'),
]
