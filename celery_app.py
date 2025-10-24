from __future__ import annotations
import os
from celery import Celery

REDIS_URL = os.getenv('REDIS_URL', '')
app = Celery('don_research_qac', broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(task_serializer='json', accept_content=['json'], result_serializer='json', timezone='UTC')

# You can wire celery tasks to call the same run_fit/run_apply functions.
# For now, the FastAPI background path is sufficient; add Celery workers when scaling.