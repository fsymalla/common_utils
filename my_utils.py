import yaml
from yaml import CLoader


def load_dict(fname):
    with open(fname,'r') as inf:
        data = yaml.load(inf,Loader=CLoader)
    return data

def print_keys(d, indent=0):
    """
    Recursively prints the hierarchy of all keys in a nested dictionary.
    """
    if not isinstance(d, dict):
        print(" " * indent + "- [Non-Dict Value]")
        return
    
    for key, value in d.items():
        print(" " * indent + f"- {key}")
        if isinstance(value, dict):
            print_keys(value, indent + 4)
        elif isinstance(value, list):
            print(" " * (indent + 4) + "- [List]")
            for i, item in enumerate(value):
                print(" " * (indent + 6) + f"[{i}]")
                if isinstance(item, dict):
                    print_keys(item, indent + 8)
        else:
            print(" " * (indent + 4) + "- [Value]")


