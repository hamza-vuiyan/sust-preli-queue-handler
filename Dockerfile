# Use the official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy requirements first to leverage Docker caching layers
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy your app directory into the container
COPY ./app /code/app

# CRITICAL: Hugging Face Spaces natively expose and expect port 7860
EXPOSE 7860

# Start Uvicorn pointing to port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]