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
import re

# def home(request):
#     data = pd.read_excel('fypdata.xlsx')
#     data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']
#     data = data.drop(0)
    
#     crime_years = ['2019', '2020', '2021', '2022', '2023', '2024']
#     area_crime_data = data.groupby('Area')[crime_years].sum()
#     area_crime_data_dict = area_crime_data.to_dict(orient='index')

#     max_crimes = 0
#     city_with_max_crimes = ""
#     year_with_max_crimes = ""
#     yearly_totals = {year: 0 for year in crime_years}

#     for city, crimes in area_crime_data_dict.items():
#         city_total = sum(crimes.values())
#         if city_total > max_crimes:
#             max_crimes = city_total
#             city_with_max_crimes = city

#         for year, count in crimes.items():
#             yearly_totals[year] += count

#     year_with_max_crimes = max(yearly_totals, key=yearly_totals.get)

#     if request.method == "POST":
#         if "message" in request.POST:
#             name = request.POST.get("name")
#             email = request.POST.get("email")
#             subject = request.POST.get("subject")
#             message = request.POST.get("message")

#             ContactMessage.objects.create(name=name, email=email, subject=subject, message=message)
#             messages.success(request, "Your message has been sent. Thank you!")

#         elif "email" in request.POST:
#             email = request.POST.get("email")
#             if email:
#                 if not NewsletterSubscription.objects.filter(email=email).exists():
#                     NewsletterSubscription.objects.create(email=email)
#                     messages.success(request, "Your subscription request has been sent. Thank you!")
#                 else:
#                     messages.warning(request, "You are already subscribed.")
#             else:
#                 messages.error(request, "Please enter a valid email.")

#         return redirect("home")

#     return render(request, "home.html", {
#         'area_crime_data': area_crime_data_dict,
#         'max_crimes': max_crimes,
#         'city_with_max_crimes': city_with_max_crimes,
#         'year_with_max_crimes': year_with_max_crimes,
#         'yearly_totals': yearly_totals
#     })

def home(request):
    data = pd.read_excel('fypdata.xlsx')
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']
    data = data.drop(0)

    crime_years = ['2019', '2020', '2021', '2022', '2023', '2024']
    area_crime_data = data.groupby('Area')[crime_years].sum()

    # Convert crime counts to integers for compatibility with widthratio
    area_crime_data_dict = {
        area: {year: int(crimes[year]) for year in crime_years}
        for area, crimes in area_crime_data.to_dict(orient='index').items()
    }

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
    if user_year < 2025 or user_year > 2030:
        return render(request, 'predict_result.html', {'error': 'Invalid year. Please enter a year between 2025 and 2030.'})

    data = pd.read_excel('fypdata.xlsx')
    data = data.drop(0)
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']
    
    train_years = np.array([2019, 2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
    predict_years = np.array([user_year]).reshape(-1, 1)

    actual_2024 = {}
    predicted_year = {}

    for _, row in data.iterrows():
        y = row[['2019', '2020', '2021', '2022', '2023', '2024']].values.astype(float).reshape(-1, 1)
        model = LinearRegression()
        model.fit(train_years, y)
        prediction = model.predict(predict_years).flatten()[0]
        category = row['Category']
        actual_2024[category] = float(row['2024'])
        predicted_year[category] = prediction

    categories = list(actual_2024.keys())
    actual_vals = [actual_2024[cat] for cat in categories]
    predicted_vals = [predicted_year[cat] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, actual_vals, width, label='2024 (Actual)', color='orange')
    plt.bar(x + width/2, predicted_vals, width, label=f'{user_year} (Predicted)', color='skyblue')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.xlabel("Crime Categories")
    plt.ylabel("Crime Count")
    plt.title(f"Crime Comparison: 2024 vs {user_year}")
    plt.legend()
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'predict_result.html', {'graph': graph, 'user_year': user_year})
def chat(request):
    return render(request, 'chat.html')
def chatbot_response(request):
    user_message = request.GET.get('message', '').lower()
    data = pd.read_excel('fypdata.xlsx')
    data = data.drop(0)
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']

    reply = "I'm sorry, I don't understand."

    # 1. Offense code lookup
    for _, row in data.iterrows():
        if row['Category'].lower() in user_message:
            reply = f"The person will be charged with the offense code {row['Offense_Code']}."
            break

    # 2. Most common crimes in [area]
    area_match = re.search(r'most common crimes in ([a-zA-Z ]+)', user_message)
    if area_match:
        area = area_match.group(1).strip().lower()
        area_data = data[data['Area'].str.lower() == area]
        top_crimes = (
            area_data.groupby('Category')[['2019', '2020', '2021', '2022', '2023', '2024']].sum()
            .sum(axis=1).sort_values(ascending=False).head(3).index.tolist()
        )
        reply = f"In {area.title()}, the most reported crimes are {', '.join(top_crimes)}."

    # 3. Crime count in 2024 in [area]
    crime_count_match = re.search(r'how many crimes happened in ([a-zA-Z ]+).*last year', user_message)
    if crime_count_match:
        area = crime_count_match.group(1).strip().lower()
        area_data = data[data['Area'].str.lower() == area]
        total_2024 = area_data['2024'].astype(int).sum()
        reply = f"In 2024, there were {total_2024} reported crimes in {area.title()}."

    # 4. Crime rate in [area] (mocked with placeholder population)
    crime_rate_match = re.search(r'crime rate in ([a-zA-Z ]+)', user_message)
    if crime_rate_match:
        area = crime_rate_match.group(1).strip().lower()
        area_data = data[
        (data['Area'].str.lower() == area) | 
        (data['Locality'].str.lower() == area)
        ]

        total_crimes = area_data[['2019', '2020', '2021', '2022', '2023', '2024']].astype(int).sum().sum()
        population = 50000  # placeholder, should be replaced with actual data
        rate = round((total_crimes / population) * 1000, 2)
        reply = f"The crime rate in {area.title()} is {rate} crimes per 1,000 residents."

    return JsonResponse({'response': reply})

def area_crime_heatmap(request):
    data = pd.read_excel('fypdata.xlsx')
    data = data.drop(0)
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']
    
    images = []

    # Heatmap 1: Area vs Year
    area_crime = data.groupby('Area')[['2019', '2020', '2021', '2022', '2023', '2024']].sum()
    plt.figure(figsize=(14, 10))
    sns.heatmap(area_crime, annot=True, cmap="mako", linewidths=0.6, fmt='g', cbar_kws={'label': 'Crime Count'})
    plt.title('Crimes per Area Over Years (2019–2024)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Area', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    buffer.close()

    # Heatmap 2: Category vs Year
    cat_year = data.groupby('Category')[['2019', '2020', '2021', '2022', '2023', '2024']].sum()
    plt.figure(figsize=(16, 12))
    sns.heatmap(cat_year, annot=True, cmap="flare", linewidths=0.6, fmt='g', cbar_kws={'label': 'Crime Count'})
    plt.title('Crimes per Category Over Years (2019–2024)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    buffer.close()

    # Heatmap 3: Area vs Category
    data['Total'] = data[['2019', '2020', '2021', '2022', '2023', '2024']].sum(axis=1)
    area_cat = data.groupby(['Area', 'Category'])['Total'].sum().unstack(fill_value=0)
    plt.figure(figsize=(18, 14))
    sns.heatmap(area_cat, annot=True, cmap="rocket_r", linewidths=0.6, fmt='g', cbar_kws={'label': 'Total Crimes'})
    plt.title('Total Crimes per Area by Category (2019–2024)', fontsize=16)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Area', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    buffer.close()

    return render(request, 'area_crime_heatmap.html', {'graphs': images})


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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import io
import base64
import seaborn as sns

def predicted_crime_by_area_view(request):
    data = pd.read_excel('fypdata.xlsx')
    data = data.drop(0)
    data.columns = ['Category', 'Offense_Code', '2019', '2020', '2021', '2022', '2023', '2024', 'Area', 'Locality']

    years = np.array([2019, 2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
    future_years = np.array([2025, 2026, 2027, 2028, 2029, 2030])
    predictions = {}

    graphs = []

    for area in data['Area'].unique():
        area_data = data[data['Area'] == area]
        for crime in area_data['Category'].unique():
            row = area_data[(area_data['Category'] == crime)]
            X = years
            y = row[['2019', '2020', '2021', '2022', '2023', '2024']].values.flatten().reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(future_years.reshape(-1, 1)).flatten()
            y_2024 = float(row['2024'].values[0])

            x = np.arange(len(future_years))
            width = 0.4

            colors_2024 = sns.color_palette("pastel", len(future_years))
            colors_pred = sns.color_palette("dark", len(future_years))

            plt.figure(figsize=(8, 4))
            for i in range(len(future_years)):
                plt.bar(x[i] - width/2, y_2024, width=width, color=colors_2024[i], label='2024' if i == 0 else "")
                plt.bar(x[i] + width/2, y_pred[i], width=width, color=colors_pred[i], label='Predicted' if i == 0 else "")
                
                plt.text(x[i] - width/2, y_2024 + y_2024 * 0.01, f'{int(y_2024)}',
                         ha='center', va='bottom', fontsize=8, fontweight='bold')
                plt.text(x[i] + width/2, y_pred[i] + y_pred[i] * 0.01, f'{int(y_pred[i])}',
                         ha='center', va='bottom', fontsize=8, fontweight='bold')

            plt.xticks(x, future_years)
            plt.title(f"{crime} in {area}: 2024 vs Predictions (2025–2030)")
            plt.xlabel("Year")
            plt.ylabel("Crime Count")
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()

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

    if nrows == 1:
        axes = [axes]

    colors = sns.color_palette("tab10", len(data['Area'].unique()))

    for i, crime in enumerate(data['Category'].unique()):
        crime_data = data[data['Category'] == crime].groupby('Area')[['2019', '2020', '2021', '2022', '2023', '2024']].sum().reset_index()
        crime_data_melted = crime_data.melt(id_vars='Area', var_name='Year', value_name='Offenses')

        for j, area in enumerate(crime_data['Area'].unique()):
            area_data = crime_data_melted[crime_data_melted['Area'] == area]
            sns.lineplot(x='Year', y='Offenses', data=area_data, marker='o', ax=axes[i], label=area, color=colors[j])

            for x, y in zip(area_data['Year'], area_data['Offenses']):
                axes[i].text(x, y + y * 0.02, f'{int(y)}', ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors[j])


        axes[i].set_title(f"Crime Offenses by Area for {crime} (2019-2024)")
        axes[i].set_xticks(['2019', '2020', '2021', '2022', '2023', '2024'])
        axes[i].set_xticklabels(['2019', '2020', '2021', '2022', '2023', '2024'], rotation=45, ha='right')
        axes[i].set_ylabel('Offenses')
        axes[i].legend(title='Area')

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'crime_rate_by_area.html', {'graph': graph})