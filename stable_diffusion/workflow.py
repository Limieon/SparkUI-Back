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


class Workflow:
    def __init__(self, nodes: dict[str, WorkflowNodeData]):
        self.nodes = nodes

    def has_recursion(self) -> bool:
        # TODO implement recursion checking
        return False

    async def invoke(self, parameters: dict[str, Any]):
        # TODO implement workflow invocation

        print(self.nodes)
        print(parameters)

        return

    nodes: dict[str, WorkflowNodeData]
