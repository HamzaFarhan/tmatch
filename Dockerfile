FROM python:3.11
WORKDIR /opt/app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /opt/app

EXPOSE 8000

ENTRYPOINT ["uvicorn","app.main:app","--reload"]

# docker build . -f Dockerfile --tag tmatch.v1
# docker run -it --rm --name tmatch -p 8000:8000 tmatch.v1