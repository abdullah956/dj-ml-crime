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
    
    images = []

    # Heatmap 1: Area vs Year
    area_crime = data.groupby('Area')[['2019', '2020', '2021', '2022', '2023', '2024']].sum()
    plt.figure(figsize=(12, 8))
    sns.heatmap(area_crime, annot=True, cmap="YlGnBu", linewidths=0.5, fmt='g')
    plt.title('Total Crimes in Areas (2019–2024)')
    plt.xlabel('Year')
    plt.ylabel('Area')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    buffer.close()

    # Heatmap 2: Category vs Year
    cat_year = data.groupby('Category')[['2019', '2020', '2021', '2022', '2023', '2024']].sum()
    plt.figure(figsize=(14, 10))
    sns.heatmap(cat_year, annot=True, cmap="OrRd", linewidths=0.5, fmt='g')
    plt.title('Total Crimes per Category (2019–2024)')
    plt.xlabel('Year')
    plt.ylabel('Category')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    buffer.close()

    # Heatmap 3: Area vs Category (Total)
    data['Total'] = data[['2019', '2020', '2021', '2022', '2023', '2024']].sum(axis=1)
    area_cat = data.groupby(['Area', 'Category'])['Total'].sum().unstack(fill_value=0)
    plt.figure(figsize=(16, 12))
    sns.heatmap(area_cat, annot=True, cmap="coolwarm", linewidths=0.5, fmt='g')
    plt.title('Total Crimes per Area per Category')
    plt.xlabel('Category')
    plt.ylabel('Area')
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
    future_years = np.array([2025, 2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
    predictions = {}

    # Define color palette (you can use seaborn color palettes for variety)
    colors = sns.color_palette("Set1", len(data['Category'].unique()))
    crime_category_to_color = {category: colors[i] for i, category in enumerate(data['Category'].unique())}

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

            # Assign a unique color for each crime category using the dictionary
            color = crime_category_to_color[crime]
            
            # Plotting the bar chart
            plt.bar(future_years.flatten(), predicted_values, color=color)
            plt.title(f"Predicted Crime Rates for {crime} in {area} (2025-2030)", fontsize=14)
            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Predicted Crime Count", fontsize=12)
            plt.xticks(future_years.flatten())

            # Adding gridlines and customizing the appearance
            plt.grid(axis='y', linestyle='--', alpha=0.7)
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

    # Generate a unique color for each area
    colors = sns.color_palette("tab10", len(data['Area'].unique()))

    for i, crime in enumerate(data['Category'].unique()):
        crime_data = data[data['Category'] == crime].groupby('Area')[['2019', '2020', '2021', '2022', '2023', '2024']].sum().reset_index()
        crime_data_melted = crime_data.melt(id_vars='Area', var_name='Year', value_name='Offenses')
        
        # Assign colors to each area line
        for j, area in enumerate(crime_data['Area'].unique()):
            area_data = crime_data_melted[crime_data_melted['Area'] == area]
            sns.lineplot(x='Year', y='Offenses', data=area_data, marker='o', ax=axes[i], label=area, color=colors[j])

        axes[i].set_title(f"Crime Offenses by Area for {crime} (2019-2024)")
        axes[i].set_xticks(['2019', '2020', '2021', '2022', '2023', '2024'])  # Ensure all years are shown on the x-axis
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