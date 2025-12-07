FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY rl_agent.py lstm_predictor.py orchestrator.py /app/

CMD ["python", "orchestrator.py"]
