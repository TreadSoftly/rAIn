# projects/drone-vision/lambda/Dockerfile
FROM public.ecr.aws/lambda/python:3.10

# ----- Copy code + model ----------------------------------------------------
#   Everything under lambda/ goes to /var/task  (= Lambda “root”)
COPY . .

# ----- Install Python deps ---------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ----- Set handler ----------------------------------------------------------
CMD ["app.handler"]
