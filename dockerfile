# Dockerfile

FROM python:3.11-slim

# Install locale packages for proper character encoding
RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Create a working directory inside the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies
# This is done early to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your application code and static/templates files
# This includes app.py, model.onnx, and the 'templates' folder
COPY . .

# Expose the port your Flask app will run on (10000)
EXPOSE 10000

# Start the application using Gunicorn
# 'app:app' means look for the Flask app instance named 'app'
# within the 'app.py' file.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "1"]