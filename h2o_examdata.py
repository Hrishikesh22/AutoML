import h2o
from pathlib import Path
from h2o_model import run_h2o_automl_on_parquet

if __name__ == "__main__":
    h2o.init(max_mem_size="2G", nthreads=-1)
    run_h2o_automl_on_parquet(Path("exam_dataset/1"), log_file="exam_dataset_log.txt")
    h2o.shutdown(prompt=False)