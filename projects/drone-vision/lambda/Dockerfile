FROM public.ecr.aws/lambda/python:3.10

# Copy function code
COPY app.py requirements.txt ./

# Install dependencies
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD ["app.handler"]
