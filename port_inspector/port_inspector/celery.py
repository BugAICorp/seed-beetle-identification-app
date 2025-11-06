""" celery.py """

import os
import sys
from celery import Celery

# Get the parent directory of the *inner* port_inspector/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'port_inspector.settings')

app = Celery('port_inspector')

# Load config from Django settings, using the CELERY_ prefix
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks in each Django app
app.autodiscover_tasks()
