.PHONY: setup preprocess train telemetry app run-all test clean

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

preprocess:
	cd project1_ml && python preprocess.py

train:
	cd project1_ml && python train_uq.py

telemetry:
	cd project2_telemetry && python pipeline.py

app:
	cd project2_telemetry && streamlit run app.py

run-all: preprocess train telemetry app

test:
	pytest -q

clean:
	rm -rf project1_ml/reports
	rm -rf project1_ml/processed/*.parquet
	rm -f project2_telemetry/cleaned_telemetry.csv
	rm -f project2_telemetry/model_predictions.csv
	rm -f project2_telemetry/model_metrics.csv
	rm -f project2_telemetry/feature_importance_top20.csv

.PHONY: demo
demo: preprocess train telemetry
	python scripts/export_demo.py
