import os, uuid, datetime as dt, sys
from google.cloud import bigquery
import requests

def log(msg):
    print(msg); sys.stdout.flush()

def run():
    project_id = os.environ.get("PROJECT_ID", "tradedesk17")
    dataset = os.environ.get("BQ_DATASET", "trading")
    discord = os.environ.get("DISCORD_WEBHOOK")

    log("weekend: start")
    run_id = str(uuid.uuid4())
    started_at = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

    log("weekend: init bigquery client")
    bq = bigquery.Client(project=project_id)  # location not required for inserts

    table_id = f"{project_id}.{dataset}.weekend_runs"
    log(f"weekend: get table {table_id}")
    table = bq.get_table(table_id)  # ensure exists & get schema

    # Use insert_rows with tuples -> BQ client serializes TIMESTAMP correctly
    rows_to_insert = [(run_id, started_at, "weekend agent placeholder run")]

    log("weekend: inserting row")
    errors = bq.insert_rows(table, rows_to_insert)  # returns a list of errors
    if errors:
        log(f"weekend: insert returned errors: {errors}")
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
