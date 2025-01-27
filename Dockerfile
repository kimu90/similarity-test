# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    postgresql-contrib \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Create data directory
RUN mkdir -p /app/data

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Expose the port Streamlit will run on
EXPOSE 8501

# Define environment variable to prevent writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# Define environment variable to prevent buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# Run the application
CMD ["streamlit", "run", "web/dashboard/app.py"]