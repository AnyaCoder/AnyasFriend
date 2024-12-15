from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PropertyItem(BaseModel):
    type: str
    description: str
    items: Optional[Dict[str, Any]] = None


class ToolParameter(BaseModel):
    type: str
    properties: Dict[str, PropertyItem]
    required: List[str]
    additionalProperties: bool = False


class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: ToolParameter


class Tool(BaseModel):
    type: str
    function: ToolFunction


predefined_tools: List[Tool] = [
    Tool(
        type="function",
        function=ToolFunction(
            name="set_emotion",
            description="根据你说的话设定一个总体情绪，一般情况下用normal，情绪比较明显的时候才能用其他词。",
            parameters=ToolParameter(
                type="object",
                properties=dict(
                    emotion=PropertyItem(
                        type="string",
                        description="情绪词，从以下列表中选择：["
                        "normal, happy, sad, "
                        "angry, quiet, speechless, "
                        "surprised, scared"
                        "]",
                    )
                ),
                required=["emotion"],
            ),
        ),
    ),
]


def func_call_main():
    tools_example = [
        {
            "type": "function",
            "function": {
                "name": "generate_recipe",
                "description": "Generate a recipe based on the user's input",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the recipe.",
                        },
                        "ingredients": {
                            "type": "array",
                            "description": "List of ingredients required for the recipe.",
                            "items": {
                                "type": "string",
                                "description": "Each ingredient in the recipe.",
                            },
                        },
                        "instructions": {
                            "type": "array",
                            "description": "Step-by-step instructions for the recipe.",
                            "items": {
                                "type": "string",
                                "description": "Each step in the recipe.",
                            },
                        },
                    },
                    "required": ["title", "ingredients", "instructions"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    # 使用Pydantic模型验证
    tool_objects = [Tool(**tool) for tool in tools_example]

    # 输出验证后的工具对象
    for tool in tool_objects:
        print(tool.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    func_call_main()
