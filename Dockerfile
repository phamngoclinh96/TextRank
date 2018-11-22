FROM python:3.6-alpine
VOLUME /app
COPY . /app/
WORKDIR /app/
RUN ls
RUN apk add --no-cache --virtual .build-deps gcc musl-dev
RUN pip install -r requirements.txt
RUN apk del .build-deps gcc musl-dev
CMD python main.py