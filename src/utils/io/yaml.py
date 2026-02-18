from typing import Type, TypeVar, Any
from cattrs import structure, unstructure
import yaml

T = TypeVar('T')

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def load_from_yaml(path: str, cl: Type[T]) -> T:
    """Loads data from a YAML file and structures it into the given class.

    Uses PyYAML for loading and cattrs for structuring the data into
    dataclasses (or other types).

    Args:
        path: The file path to the YAML file to load.
        cl: The target type (e.g., a dataclass, List[dataclass])
            to structure the data into.

    Returns:
        An instance of the target type (T) populated with data
        from the YAML file.
        
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        yaml.YAMLError: If the file content is not valid YAML.
        cattrs.errors.StructureError: If the YAML data cannot be
            structured into the target class 'cl'.
    """
    with open(path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
        data_structured = structure(yaml_data, cl)
    return data_structured
        

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def save_as_yaml(path: str, data: Any):
    """Saves a data object (e.g., dataclass) to a YAML file.

    Uses cattrs to unstructure the object into basic Python types
    (dicts, lists) and PyYAML to dump it to a file.

    The output is formatted for readability:
    - Keys are not sorted alphabetically (preserves dataclass order).
    - Block style is used (no inline {} or []).
    - Unicode characters are preserved (not escaped).

    Args:
        path: The destination file path to write the YAML data to.
        data: The data object to save (e.g., a dataclass instance,
            a list of dataclasses, etc.).
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml_data = unstructure(data)
        yaml.dump(
            yaml_data,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True
        )