# Use an official Python image
FROM python:3.12.3

# Set working directory
WORKDIR /Quant

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the script
CMD ["python", "Optimization/FAPT_Optimization.py"]
