# Specify your base image
FROM python:3.8
# create a work directory
RUN mkdir /app
# navigate to this work directory
WORKDIR /app
#Copy all files
COPY . .
# Install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
# Run
CMD ["python3","audio2text.py"]