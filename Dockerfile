FROM python:3.12-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/

# Persistence lives here: voice bookings, saved quotes, custom personas, the
# run log and published memos. Mount a volume at /app/data in compose or every
# custom agent the builder mints is lost on redeploy.
ENV OBS_DATA_DIR=/app/data
VOLUME ["/app/data"]

EXPOSE 8321
CMD ["python", "-m", "uvicorn", "app:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8321"]
