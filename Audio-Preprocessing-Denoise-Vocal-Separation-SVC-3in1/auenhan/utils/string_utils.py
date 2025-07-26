import random
import string

def generate_random_string(length:int):
    """
    Generate a random string of specified length.

    Args:
        length (int): The length of the random string.

    Returns:
        str: The generated random string.
    """
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))

