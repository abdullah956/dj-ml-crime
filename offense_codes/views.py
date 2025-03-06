import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import urllib, base64
import numpy as np
from sklearn.linear_model import LinearRegression
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import ContactMessage, NewsletterSubscription

def home(request):
    data = pd.read_excel('fypdata.xlsx')
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']
    data = data.drop(0)
    
    crime_years = ['2019', '2020', '2021', '2022', '2023', '2024']
    area_crime_data = data.groupby('Area')[crime_years].sum()
    area_crime_data_dict = area_crime_data.to_dict(orient='index')

    max_crimes = 0
    city_with_max_crimes = ""
    year_with_max_crimes = ""
    yearly_totals = {year: 0 for year in crime_years}

    for city, crimes in area_crime_data_dict.items():
        city_total = sum(crimes.values())
        if city_total > max_crimes:
            max_crimes = city_total
            city_with_max_crimes = city

        for year, count in crimes.items():
            yearly_totals[year] += count

    year_with_max_crimes = max(yearly_totals, key=yearly_totals.get)

    if request.method == "POST":
        if "message" in request.POST:
            name = request.POST.get("name")
            email = request.POST.get("email")
            subject = request.POST.get("subject")
            message = request.POST.get("message")

            ContactMessage.objects.create(name=name, email=email, subject=subject, message=message)
            messages.success(request, "Your message has been sent. Thank you!")

        elif "email" in request.POST:
            email = request.POST.get("email")
            if email:
                if not NewsletterSubscription.objects.filter(email=email).exists():
                    NewsletterSubscription.objects.create(email=email)
                    messages.success(request, "Your subscription request has been sent. Thank you!")
                else:
                    messages.warning(request, "You are already subscribed.")
            else:
                messages.error(request, "Please enter a valid email.")

        return redirect("home")

    return render(request, "home.html", {
        'area_crime_data': area_crime_data_dict,
        'max_crimes': max_crimes,
        'city_with_max_crimes': city_with_max_crimes,
        'year_with_max_crimes': year_with_max_crimes,
        'yearly_totals': yearly_totals
    })

def search_area(request):
    return render(request, 'search_area.html')

def search_result(request):
    area_input = request.GET.get('area') 
    data = pd.read_excel('fypdata.xlsx')
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']
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
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']
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
    user_message = request.GET.get('message', '').lower()
    data = pd.read_excel('fypdata.xlsx')
    data_cleaned = data[['Unnamed: 0', 'Unnamed: 1']]
    data_cleaned.columns = ['Category', 'Offense_Code']
    reply = "I'm sorry, I don't understand."
    for _, row in data_cleaned.iterrows():
        category = row['Category'].lower()
        if category in user_message:
            offense_code = row['Offense_Code']
            reply = f"The person will be charged with the offense code {offense_code}."
            break
    return JsonResponse({'response': reply})


def area_crime_heatmap(request):
    data = pd.read_excel('fypdata.xlsx')
    data = data.drop(0)
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']
    area_crime_data = data.groupby('Area')[['2019', '2020', '2021', '2022', '2023', '2024']].sum()
    plt.figure(figsize=(12, 8))
    sns.heatmap(area_crime_data, annot=True, cmap="YlGnBu", linewidths=0.5, fmt='g')
    plt.title('Total Crimes in Areas (2019-2024)')
    plt.xlabel('Year')
    plt.ylabel('Area')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph = base64.b64encode(image_png).decode('utf-8')
    return render(request, 'area_crime_heatmap.html', {'graph': graph})


# def predicted_crime_by_area_view(request):
#     data = pd.read_excel('fypdata.xlsx')
#     data = data.drop(0)
#     data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']
    
#     years = np.array([2019, 2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
#     future_years = np.array([2025, 2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
#     predictions = {}

#     for area in data['Area'].unique():
#         area_data = data[data['Area'] == area]
#         for crime_category in area_data['Category'].unique():
#             crime_data = area_data[area_data['Category'] == crime_category]
#             crime_values = crime_data[['2019', '2020', '2021', '2022', '2023', '2024']].values.flatten().reshape(-1, 1)
#             model = LinearRegression()
#             model.fit(years, crime_values)
#             future_predictions = model.predict(future_years)
            
#             if area not in predictions:
#                 predictions[area] = {}
#             predictions[area][crime_category] = future_predictions.flatten()

#     graphs = []
#     for area, area_predictions in predictions.items():
#         for crime, predicted_values in area_predictions.items():
#             plt.figure(figsize=(8, 4))
#             plt.bar(future_years.flatten(), predicted_values, color='skyblue')
#             plt.title(f"Predicted Crime Rates for {crime} in {area} (2025-2030)")
#             plt.xlabel("Year")
#             plt.ylabel("Predicted Crime Count")
#             plt.xticks(future_years.flatten())

#             buffer = io.BytesIO()
#             plt.savefig(buffer, format='png')
#             buffer.seek(0)
#             image_png = buffer.getvalue()
#             buffer.close()
#             graph = base64.b64encode(image_png).decode('utf-8')
#             graphs.append((area, crime, graph))
#             plt.close()

#     return render(request, 'predicted_crime_by_area.html', {'graphs': graphs})

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

def predicted_crime_by_area_view(request):
    data = pd.read_excel('fypdata.xlsx')
    data = data.drop(0)
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']
    
    years = np.array([2019, 2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
    future_years = np.array([2025, 2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
    predictions = {}

    for area in data['Area'].unique():
        area_data = data[data['Area'] == area]
        for crime_category in area_data['Category'].unique():
            crime_data = area_data[area_data['Category'] == crime_category]
            crime_values = crime_data[['2019', '2020', '2021', '2022', '2023', '2024']].values.flatten().reshape(-1, 1)
            model = LinearRegression()
            model.fit(years, crime_values)
            future_predictions = model.predict(future_years)
            
            if area not in predictions:
                predictions[area] = {}
            predictions[area][crime_category] = future_predictions.flatten()

    graphs = []
    for area, area_predictions in predictions.items():
        for crime, predicted_values in area_predictions.items():
            plt.figure(figsize=(8, 4))
            plt.bar(future_years.flatten(), predicted_values, color='skyblue')
            plt.title(f"Predicted Crime Rates for {crime} in {area} (2025-2030)")
            plt.xlabel("Year")
            plt.ylabel("Predicted Crime Count")
            plt.xticks(future_years.flatten())

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graph = base64.b64encode(image_png).decode('utf-8')
            graphs.append((area, crime, graph))
            plt.close()

    return render(request, 'predicted_crime_by_area.html', {'graphs': graphs})



def crime_rate_by_area_view(request):
    data = pd.read_excel('fypdata.xlsx')
    data = data.drop(0)
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']

    ncols = 1
    nrows = len(data['Category'].unique())

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6 * nrows))

    for i, crime in enumerate(data['Category'].unique()):
        crime_data = data[data['Category'] == crime].groupby('Area')[['2019', '2020', '2021', '2022', '2023', '2024']].sum().reset_index()
        crime_data_melted = crime_data.melt(id_vars='Area', var_name='Year', value_name='Offenses')
        sns.barplot(x='Area', y='Offenses', hue='Year', data=crime_data_melted, palette='coolwarm', ax=axes[i])
        axes[i].set_title(f"Crime Offenses by Area for {crime} (2019-2024)")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph = base64.b64encode(image_png).decode('utf-8')
    
    return render(request, 'crime_rate_by_area.html', {'graph': graph})
