"""
Helper functions for bio module tests
"""

def create_repeated_labels(label_list, n_cells):
    """
    Create repeated label array that exactly matches n_cells
    
    Args:
        label_list: List of labels to repeat (e.g., ['0', '1'])
        n_cells: Total number of cells needed
        
    Returns:
        List of exactly n_cells labels
    """
    full_reps = n_cells // len(label_list)
    remainder = n_cells % len(label_list)
    
    labels = label_list * full_reps
    labels += label_list[:remainder]
    
    return labels[:n_cells]
