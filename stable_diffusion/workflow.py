from pydantic import BaseModel
from typing import Any, Union


class WorkflowNodeInputData_Value(BaseModel):
    type: str = "value"
    value: Any


class WorkflowNodeInputData_Node_Data(BaseModel):
    id: str
    handle: str


class WorkflowNodeInputData_Node(BaseModel):
    type: str = "node"
    value: WorkflowNodeInputData_Node_Data


class WorkflowNodeInputData_Parameter(BaseModel):
    type: str = "parameter"
    value: str


class WorkflowNodeData(BaseModel):
    type: str
    variant: str
    inputs: list[Union[WorkflowNodeInputData_Node, WorkflowNodeInputData_Parameter, WorkflowNodeInputData_Value]]


class WorkflowData(BaseModel):
    nodes: dict[str, WorkflowNodeData]
    parameters: dict[str, Any]
