cmake_minimum_required(VERSION 3.10)

# Project name
project(TransferRL)

# Path to your virtual environment
set(VENV_PATH "${CMAKE_SOURCE_DIR}/myenv")

# Python interpreter within the virtual environment
set(PYTHON_EXECUTABLE "${VENV_PATH}/bin/python")

set(SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/scripts")

# Apply PYTHONPATH to all script targets
set(ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}")

# Run (trainBaseModel.py)
add_custom_target(RunTBM 
    COMMAND PYTHONPATH=${CMAKE_SOURCE_DIR} ${PYTHON_EXECUTABLE} ${SCRIPTS_DIR}/trainBaseModel.py 
    COMMENT "Running trainBaseModel.py"
)

# Run (trainTransferModel.py)
add_custom_target(RunTTM 
    COMMAND PYTHONPATH=${CMAKE_SOURCE_DIR} ${PYTHON_EXECUTABLE} ${SCRIPTS_DIR}/trainTransferModel.py 
    COMMENT "Running trainTransferModel.py"
)

# Run (fineToonModel.py)
add_custom_target(RunTFM 
    COMMAND PYTHONPATH=${CMAKE_SOURCE_DIR} ${PYTHON_EXECUTABLE} ${SCRIPTS_DIR}/fineToonModel.py 
    COMMENT "Running fineToonModel.py"
)

# Run (evaluateBaseModel.py)
add_custom_target(RunEBM 
    COMMAND PYTHONPATH=${CMAKE_SOURCE_DIR} ${PYTHON_EXECUTABLE} ${SCRIPTS_DIR}/evaluateBaseModel.py 
    COMMENT "Running evaluateBaseModel.py"
)

# Run (evaluateTransferModel.py)
add_custom_target(RunETM 
    COMMAND PYTHONPATH=${CMAKE_SOURCE_DIR} ${PYTHON_EXECUTABLE} ${SCRIPTS_DIR}/evaluateTransferModel.py 
    COMMENT "Running evaluateTransferModel.py"
)

# Run (evaluateFineToonedModel.py)
add_custom_target(RunEFM 
    COMMAND PYTHONPATH=${CMAKE_SOURCE_DIR} ${PYTHON_EXECUTABLE} ${SCRIPTS_DIR}/evaluateFineToonedModel.py 
    COMMENT "Running evaluateFineToonedModel.py"
)


