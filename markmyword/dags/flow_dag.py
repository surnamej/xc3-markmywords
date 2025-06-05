import os
import sys
from datetime import datetime, timedelta
# from airflow import DAG
# from airflow.operators.python import PythonOperator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.phase1 import phase1
from pipelines.phase2 import phase2

def test_function():
    # phase1()
    phase2()
if __name__ == "__main__":
    test_function()