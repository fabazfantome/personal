import os.path
import requests


def get_data(path: str, filename: str, url: str) -> None:
    if os.path.exists(path) == False:
        raise RuntimeError('The selected path does not exist.')
    with open(os.path.join(path, filename), 'wt') as file:
        file.write(requests.get(url, allow_redirects = True).text)

