from django.contrib import admin
from .models import ContactMessage , NewsletterSubscription

admin.site.register(ContactMessage)
admin.site.register(NewsletterSubscription)