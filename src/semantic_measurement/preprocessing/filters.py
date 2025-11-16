def remove_factset_metadata(text):
    """
    Removes the recurring FactSet metadata block by deleting 'FactSet CallStreet, LLC' 
    and the 10 lines before it.
    """
    lines = text.split("\n")  # Split text into lines

    indices_to_remove = [i for i, line in enumerate(lines) if "FactSet CallStreet, LLC" in line]
    
    for index in reversed(indices_to_remove):  # Reverse to avoid shifting indices
        start_index = max(0, index - 10)  # Ensure we don't go below 0
        del lines[start_index:index + 2]  # Delete the block

    return "\n".join(lines)  # Reconstruct text

