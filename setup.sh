BASE_DIR=${PWD}
cd ${BASE_DIR}/microxcaling && pip install -r requirements.txt
cd ${BASE_DIR}/microxcaling && pip install -e .
cd ${BASE_DIR}/lm-evaluation-harness && pip install -e .
cd ${BASE_DIR} && pip install -r requirements.txt
cd ${BASE_DIR}
