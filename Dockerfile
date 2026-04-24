FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir jupyter build && \
    python -m build && \
    pip install dist/*.whl
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
