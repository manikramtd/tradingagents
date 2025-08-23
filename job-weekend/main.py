import os, uuid, datetime as dt, sys
from google.cloud import bigquery
import requests

def log(msg):
    print(msg); sys.stdout.flush()

def run():
    project_id = os.environ.get("PROJECT_ID")
    dataset = os.environ.get("BQ_DATASET", "trading")
    discord = os.environ.get("DISCORD_WEBHOOK")

    log("weekend: start")

    run_id = str(uuid.uuid4())
    started_at = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

    log("weekend: init bigquery client")
    bq = bigquery.Client(project=project_id, location="US")  # adjust if dataset is EU

    table = f"{project_id}.{dataset}.weekend_runs"
    row = {
        "run_id": run_id,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "notes": "weekend agent placeholder run"
    }

       log(f"weekend: inserting row into {table}")
    try:
        errors = bq.insert_rows_json(table, [row], timeout=30)
    except Exception as e:
        log(f"weekend: bq insert raised: {repr(e)}")
        raise
    if errors:
        log(f"weekend: bq insert returned row errors: {errors}")
        raise RuntimeError(f"BigQuery insert errors: {errors}")
    log("weekend: bq insert ok")


    if discord:
        try:
            log("weekend: posting to discord")
            requests.post(discord, json={
                "content": f"🧮 Weekend Agent ran OK: {run_id} @ {started_at.isoformat()} UTC"
            }, timeout=6)
            log("weekend: discord post ok")
        except Exception as e:
            log(f"weekend: discord post failed: {e}")

    log("weekend: done")

if __name__ == "__main__":
    run()
