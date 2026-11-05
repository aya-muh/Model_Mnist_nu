FROM python:3.9-slim

# Install system dependencies required by LightGBM (libgomp1)

# Set the working directory in the container
WORKDIR /Model_Mnist

# Copy requirements.txt
COPY requirements.txt .

# Install the Python dependencies
RUN \pip install --no-cache-dir --default-timeout=100 -r requirements.txt
#
#RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# Copy the entire project directory into the container
COPY . .

# Expose port 5000 for the Flask application
EXPOSE 5000

# Define the command to run the application
#CMD ["python", "Machan learning\Model_Mnist\digit_recognition_service\app.py"]
CMD ["python", "digit_recognition_service/app.py"]
