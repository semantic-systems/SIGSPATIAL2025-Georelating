def preprocess_data(data: dict, model) -> dict:
    """
    Replace None values in a dictionary with the default values defined in a Pydantic model.

    Args:
        data (dict): The dictionary to preprocess.
        model: The Pydantic model defining the default values.

    Returns:
        dict: The preprocessed dictionary.
    """
    defaults = {field: field_info.default for field, field_info in model.model_fields.items() if
                field_info.default is not None}
    for key, value in data.items():
        if value is None and key in defaults:
            data[key] = defaults[key]
    return data