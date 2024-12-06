cmake_minimum_required(VERSION 3.10)

# Project name
project(TransferRL)

# Path to your virtual environment
set(VENV_PATH "${CMAKE_SOURCE_DIR}/myenv")

# Python interpreter within the virtual environment
set(PYTHON_EXECUTABLE "${VENV_PATH}/bin/python")

# # Run (trainBaseModel.py)
# add_custom_target(RunScripts ALL
#     COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/trainBaseModel.py 
#     COMMENT "Running trainBaseModel.py"
# )

# # Run (trainTransferModel.py)
# add_custom_target(RunScripts ALL
#     COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/trainTransferModel.py 
#     COMMENT "Running trainTransferModel.py"
# )

# # Run (fineToonModel.py)
# add_custom_target(RunScripts ALL
#     COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/fineToonModel.py 
#     COMMENT "Running fineToonModel.py"
# )

# # Run (evaluateBaseModel.py)
# add_custom_target(RunScripts ALL
#     COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/evaluateBaseModel.py 
#     COMMENT "Running evaluateBaseModel.py"
# )

# # Run (evaluateTransferModel.py)
# add_custom_target(RunScripts ALL
#     COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/evaluateTransferModel.py 
#     COMMENT "Running evaluateTransferModel.py"
# )

# # Run (evaluateFineToonedModel.py)
# add_custom_target(RunScripts ALL
#     COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/evaluateFineToonedModel.py 
#     COMMENT "Running evaluateFineToonedModel.py"
# )


