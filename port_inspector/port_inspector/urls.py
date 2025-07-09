"""
URL configuration for port_inspector project.

The `urlpatterns` list routes URLs to views. For more information please see:
   https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
   1. Add an import:  from my_app import views
   2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
   1. Add an import:  from other_app.views import Home
   2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
   1. Import the include() function: from django.urls import include, path
   2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from django.views.generic import RedirectView
from port_inspector_app import views

from . import settings

urlpatterns = [
    path("", views.home_view, name="home"),
    path("check_retrain_status/", views.check_retrain_status, name="check_retrain_status"),
    path("admin/run-task/", views.retrain_models_thread, name='retrain_models_thread'),
    path("admin/", admin.site.urls),
    path("upload/", views.upload_image, name="upload"),
    path("history/", views.view_history, name="history"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("signup/", views.signup_view, name="signup"),
    path("forgot-password/", views.forgot_password, name="forgot-password"),
    path("reset-password-sent/<int:user_id>/", views.reset_password_sent, name="reset-password-sent"),
    path("reset-password/<uidb64>/<token>/", views.reset_password, name="reset-password"),
    path("verify-email/<int:user_id>/", views.verify_email, name="verify-email"),
    path("verify-email/done/", views.verify_email_done, name="verify-email-done"),
    path(
        "verify-email-confirm/<uidb64>/<token>/",
        views.verify_email_confirm,
        name="verify-email-confirm",
    ),
    path("results/<str:hashed_ID>", views.results_view, name="results"), path("notify_unknown/", views.notify_unknown, name="notify_unknown"),
    path("profile/", views.profile_view, name="profile"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# storing uploaded images to our MEDIA_URL
