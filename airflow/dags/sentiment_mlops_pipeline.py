"""Airflow DAG describing the periodic MLOps lifecycle for sentiment analysis.

The tasks are intentionally lightweight: the project focuses on orchestration and
control flow rather than expensive model fine-tuning. Values returned by the
functions simulate artifacts that, in production, would come from data storage,
evaluation jobs, a drift detector, and a model registry.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

DATA_DRIFT_THRESHOLD = 0.20
MIN_MACRO_F1 = 0.80


def collect_new_data() -> dict:
    """Simulate collection of the latest social-media batch."""
    return {
        "batch_id": "weekly-social-batch",
        "samples": 500,
        "sentiment_counts": {
            "positive": 140,
            "neutral": 120,
            "negative": 240,
        },
    }


def validate_data(ti) -> dict:
    """Validate basic completeness and schema expectations."""
    batch = ti.xcom_pull(task_ids="collect_new_data")
    samples = batch["samples"]
    counts = batch["sentiment_counts"]

    if samples <= 0:
        raise ValueError("Collected batch is empty.")
    if sum(counts.values()) != samples:
        raise ValueError("Sentiment counts do not match batch size.")

    return {
        "batch_id": batch["batch_id"],
        "samples": samples,
        "quality_status": "passed",
    }


def evaluate_current_model(ti) -> dict:
    """Simulate evaluation against a labeled validation sample."""
    validation = ti.xcom_pull(task_ids="validate_data")
    return {
        "batch_id": validation["batch_id"],
        "accuracy": 0.82,
        "macro_f1": 0.79,
    }


def detect_drift(ti) -> dict:
    """Compute a simple distribution drift score versus a reference window."""
    batch = ti.xcom_pull(task_ids="collect_new_data")
    counts = batch["sentiment_counts"]
    samples = batch["samples"]

    current = {label: value / samples for label, value in counts.items()}
    reference = {
        "positive": 0.40,
        "neutral": 0.35,
        "negative": 0.25,
    }

    # Total variation distance: 0 means identical distributions, 1 maximal shift.
    drift_score = 0.5 * sum(
        abs(current[label] - reference[label]) for label in reference
    )

    return {
        "data_drift_score": round(drift_score, 3),
        "threshold": DATA_DRIFT_THRESHOLD,
        "drift_detected": drift_score > DATA_DRIFT_THRESHOLD,
    }


def choose_retraining_path(ti) -> str:
    """Branch when drift is material or labeled performance falls below target."""
    drift = ti.xcom_pull(task_ids="detect_drift")
    evaluation = ti.xcom_pull(task_ids="evaluate_current_model")

    should_retrain = (
        drift["data_drift_score"] > DATA_DRIFT_THRESHOLD
        or evaluation["macro_f1"] < MIN_MACRO_F1
    )
    return "retrain_model" if should_retrain else "skip_retraining"


def retrain_model(ti) -> dict:
    """Represent a retraining/fine-tuning job and return a candidate version."""
    batch = ti.xcom_pull(task_ids="collect_new_data")
    return {
        "candidate_version": "sentiment-model-candidate-v2",
        "training_samples": batch["samples"],
        "status": "completed",
    }


def evaluate_candidate(ti) -> dict:
    """Simulate validation of the newly trained model candidate."""
    candidate = ti.xcom_pull(task_ids="retrain_model")
    return {
        "candidate_version": candidate["candidate_version"],
        "accuracy": 0.88,
        "macro_f1": 0.87,
        "accepted": True,
    }


def register_candidate(ti) -> dict:
    """Represent promotion to a model registry after candidate validation."""
    evaluation = ti.xcom_pull(task_ids="evaluate_candidate")
    if not evaluation["accepted"]:
        raise ValueError("Candidate model did not pass validation.")

    return {
        "registered_version": evaluation["candidate_version"],
        "stage": "candidate",
    }


with DAG(
    dag_id="sentiment_mlops_lifecycle",
    description=(
        "Periodic data collection, evaluation, drift detection " "and retraining flow"
    ),
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * 0",
    catchup=False,
    tags=["mlops", "sentiment", "drift"],
    default_args={"owner": "machineinnovators", "retries": 1},
) as dag:
    collect = PythonOperator(
        task_id="collect_new_data",
        python_callable=collect_new_data,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    evaluate = PythonOperator(
        task_id="evaluate_current_model",
        python_callable=evaluate_current_model,
    )

    drift = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift,
    )

    decide = BranchPythonOperator(
        task_id="decide_retraining",
        python_callable=choose_retraining_path,
    )

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )

    candidate_eval = PythonOperator(
        task_id="evaluate_candidate",
        python_callable=evaluate_candidate,
    )

    register = PythonOperator(
        task_id="register_candidate",
        python_callable=register_candidate,
    )

    skip = EmptyOperator(task_id="skip_retraining")

    finish = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    collect >> validate
    validate >> [evaluate, drift]
    [evaluate, drift] >> decide
    decide >> retrain >> candidate_eval >> register >> finish
    decide >> skip >> finish
