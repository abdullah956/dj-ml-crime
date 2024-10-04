from django.shortcuts import render
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import urllib, base64
import numpy as np
from sklearn.linear_model import LinearRegression
from django.http import JsonResponse

def home(request):
    data = pd.read_excel('fypdata.xlsx')
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area']
    data = data.drop(0)
    area_crime_data = data.groupby('Area')[['2019', '2020', '2021', '2022', '2023', '2024']].sum()
    area_crime_data_dict = area_crime_data.to_dict(orient='index')
    return render(request, 'home.html', {'area_crime_data': area_crime_data_dict})

def search_area(request):
    return render(request, 'search_area.html')

def search_result(request):
    area_input = request.GET.get('area') 
    data = pd.read_excel('fypdata.xlsx')
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area']
    area_data = data[data['Area'].str.lower() == area_input.lower()]
    if area_data.empty:
        return render(request, 'search_result.html', {'error': 'No data found for the specified area.'})
    heatmap_data = area_data[['Category', '2019', '2020', '2021', '2022', '2023', '2024']]
    heatmap_data.set_index('Category', inplace=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="g", cmap="YlGnBu", linewidths=0.5)
    plt.title(f"Crime Offenses Heatmap for {area_input} (2019-2024)")
    plt.xlabel("Year")
    plt.ylabel("Crime Category")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    return render(request, 'search_result.html', {'graph': graph, 'area_input': area_input})


def predict_year(request):
    return render(request, 'predict_year.html')

def predict_result(request):
    user_year = int(request.GET.get('year'))
    data = pd.read_excel('fypdata.xlsx')
    data = data.drop(0)
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area']
    years = np.array([2019, 2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
    future_years = np.array([2025, 2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
    predictions = {}
    for i, row in data.iterrows():
        crime_data = row[['2019', '2020', '2021', '2022', '2023', '2024']].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(years, crime_data)
        future_predictions = model.predict(future_years)
        predictions[row['Category']] = future_predictions.flatten()
    future_years_df = pd.DataFrame(predictions, index=[2025, 2026, 2027, 2028, 2029, 2030])

    if user_year in future_years_df.index:
        predicted_crime_rates = future_years_df.loc[user_year]
        plt.figure(figsize=(10, 6))
        plt.bar(predicted_crime_rates.index, predicted_crime_rates.values, color='skyblue')
        plt.title(f"Predicted Crime Rates for {user_year}")
        plt.xlabel("Crime Categories")
        plt.ylabel("Predicted Crime Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graph = base64.b64encode(image_png)
        graph = graph.decode('utf-8')
        return render(request, 'predict_result.html', {'graph': graph, 'user_year': user_year})
    else:
        return render(request, 'predict_result.html', {'error': 'Invalid year. Please enter a year between 2025 and 2030.'})

def chat(request):
    return render(request, 'chat.html')

def chatbot_response(request):
    user_message = request.GET.get('message').lower()

    responses = {
        "bike thief": "The person will be charged with the offense code 381a.",
        "theft": "The person will be charged with the offense code 380.",
        "girl kidnapping": "The person will be charged with the offense code 365b.",
        "electricity theft": "The person will be charged with the offense code 462j.",
        "dacoity": "The person will be charged with the offense code 392.",
        "us": "The person will be charged with arm ordinance.",
        "murder": "The person will be charged with the offense code 302.",
        "check bounce": "The person will be charged with the offense code 489f.",
        "narcotics": "The person will be charged with the offense code 3/4.",
        "fight": "The person will be charged with the offense code 337.",
        "fight with women": "The person will be charged with the offense code 354.",
        "sata bazi": "The person will be charged with gambling.",
    }

    if user_message == "hi" or user_message == "hello":
        reply = "Hello! How can I help you today?"
    else:
        reply = "I'm sorry, I don't understand."
        for key in responses:
            if key in user_message:
                reply = responses[key]
                break
    return JsonResponse({'response': reply})
