FROM python:3.8.10
WORKDIR /usr/src/
COPY requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "./app.py"]
