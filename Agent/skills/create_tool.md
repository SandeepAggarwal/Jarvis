# Skill: Create and Register a Local Agent Tool

## Purpose

Create a new callable tool for the local agent by:

1. Implementing a Python method under `./tools/`.
2. Defining the tool's JSON schema alongside the implementation.
3. Registering the tool in `./local_agent.py`.
4. Updating both the `TOOLS` array and `TOOL_FUNCTIONS` mapping.
5. Saving the modified files.

## File Structure

New tools must be created under:

```text
./tools/
```

The implementation should normally live in its own Python file, for example:

```text
./tools/tool_b.py
```

The tool file should contain:

- The Python method implementing the behavior.
- A JSON-schema-style tool definition describing how the LLM should invoke the method.

## Step 1: Implement the Tool Method

Define a Python method that accepts the required arguments and returns either:

- a `str`, or
- a JSON-compatible Python object such as a `dict` or `list`.

For example:

```python
def tool_b(argument_a: str) -> str:
    # Implement the desired behavior here.
    return "result"
```

The implementation should:

- Validate or safely handle its arguments where appropriate.
- Perform the requested operation.
- Return a useful result to the agent.
- Avoid printing the result when it should instead be returned.
- Use exceptions for unrecoverable errors so the caller can handle them.

## Step 2: Define the Tool JSON Schema

In the same tool file, define a tool configuration object.

Example:

```python
TOOL_B = {
    "type": "function",
    "function": {
        "name": "tool_b",
        "description": "Describe what the tool does and when the agent should use it.",
        "parameters": {
            "type": "object",
            "properties": {
                "argument_a": {
                    "type": "string",
                    "description": "Description of argument_a."
                }
            },
            "required": ["argument_a"]
        }
    }
}
```

### Schema Rules

- `function.name` must match the tool's callable name.
- Every accepted method argument that the LLM needs to provide should appear in `properties`.
- Each property should have the correct JSON Schema type.
- Required arguments must appear in `required`.
- Descriptions should clearly explain both the tool and its arguments.
- Do not expose implementation-only parameters that the LLM does not need to provide.

For example, if the implementation is:

```python
def fetch_webpage(url: str) -> str:
    ...
```

the corresponding schema should expose `url` as a required string argument.

## Step 3: Register the Tool in `local_agent.py`

Open:

```text
./local_agent.py
```

Locate the `_build_tool_configs` method and the existing tool configuration.

The local agent maintains two related structures:

```python
TOOLS = [
    FETCH_WEBPAGE_TOOL,
]

TOOL_FUNCTIONS = {
    "fetch_webpage": "fetch_webpage",
}
```

Add the new tool's schema to `TOOLS`:

```python
TOOLS = [
    FETCH_WEBPAGE_TOOL,
    TOOL_B,
]
```

Then add its function mapping to `TOOL_FUNCTIONS`:

```python
TOOL_FUNCTIONS = {
    "fetch_webpage": "fetch_webpage",
    "tool_b": "method_name_b",
}
```

The mapping has this form:

```text
LLM tool name -> Python method name
```

For example:

```python
"tool_b": "method_name_b"
```

means that when the LLM requests the tool named `tool_b`, the agent invokes the Python method `method_name_b`.

## Step 4: Import the New Tool

Ensure `local_agent.py` can access both the schema and the implementation method.

For example, if `./tools/tool_b.py` contains:

```python
def method_name_b(argument_a: str) -> str:
    ...
```

and:

```python
TOOL_B = {
    ...
}
```

the appropriate import should expose both:

```python
from ../tools.tool_b import TOOL_B, method_name_b
```

Follow the import conventions already used by the existing project rather than introducing a different import pattern.

## Step 5: Verify the Registration

After modifying `local_agent.py`, verify that:

- The new tool schema is included in `TOOLS`.
- The tool name is present in `TOOL_FUNCTIONS`.
- The value in `TOOL_FUNCTIONS` exactly matches the Python method name.
- The method is importable.
- The JSON schema arguments match the method's arguments.
- There are no duplicate tool names.
- Both modified files are saved.

A complete example is:

### `./tools/tool_b.py`

```python
def method_name_b(argument_a: str) -> str:
    return f"Processed: {argument_a}"


TOOL_B = {
    "type": "function",
    "function": {
        "name": "tool_b",
        "description": "Process the supplied argument.",
        "parameters": {
            "type": "object",
            "properties": {
                "argument_a": {
                    "type": "string",
                    "description": "The value to process."
                }
            },
            "required": ["argument_a"]
        }
    }
}
```

### `./local_agent.py`

```python
from ../tools.tool_b import TOOL_B, method_name_b

TOOLS = [
    FETCH_WEBPAGE_TOOL,
    TOOL_B,
]

TOOL_FUNCTIONS = {
    "fetch_webpage": "fetch_webpage",
    "tool_b": "method_name_b",
}
```

## General Procedure

When asked to create a new tool:

1. Determine the tool's name, purpose, inputs, and expected output.
2. Create a Python file under `./tools/`.
3. Implement the method.
4. Define the tool JSON schema in the same file.
5. Open `./local_agent.py`.
6. Update `_build_tool_configs` or the corresponding tool configuration section.
7. Add the schema to `TOOLS`.
8. Add the LLM-name-to-method-name mapping to `TOOL_FUNCTIONS`.
9. Add the necessary import.
10. Save all changes.
11. Verify that the method and configuration are consistent.

## Important Constraint

Do not merely describe or propose the tool.

The task is complete only after the implementation file and the corresponding `local_agent.py` registration have been created or updated and saved.
