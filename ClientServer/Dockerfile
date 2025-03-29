FROM python:3.11.11

RUN apt-get update && apt-get install -y locales \
    && sed -i 's/# ru_RU.UTF-8 UTF-8/ru_RU.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen

ENV LANG=ru_RU.UTF-8
ENV LANGUAGE=ru_RU:ru
ENV LC_ALL=ru_RU.UTF-8

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt
RUN pip uninstall jwt python-jose

CMD ["python", "main.py"]