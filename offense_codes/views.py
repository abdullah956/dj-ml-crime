from django.shortcuts import render
import pandas as pd

def home(request):
    data = pd.read_excel('fypdata.xlsx')
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area']
    data = data.drop(0)
    area_crime_data = data.groupby('Area')[['2019', '2020', '2021', '2022', '2023', '2024']].sum()
    area_crime_data_dict = area_crime_data.to_dict(orient='index')
    return render(request, 'home.html', {'area_crime_data': area_crime_data_dict})

def search_area(request):
    return render(request, 'search_area.html')

def predict_year(request):
    return render(request, 'predict_year.html')

def chat(request):
    return render(request, 'chat.html')