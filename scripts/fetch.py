import argparse
import os
import time

from bs4 import BeautifulSoup
import requests


def main(page_size: int, output_dir: str) -> None:
    """
    Fetch open access articles from the Journal of Chemical Information and
    Modeling in Europe PMC and save their text content to the specified output
    directory.

    Parameters
    ----------
    page_size : int
        Number of results to fetch from Europe PMC (default: 1000).
    output_dir : str
        Directory to save the article text files.
    """

    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": 'JOURNAL:"J Chem Inf Model" OPEN_ACCESS:Y',
        "format": "json",
        "pageSize": page_size
    }
    response = requests.get(url, params=params)
    data = response.json()
    pmcids = [result["pmcid"] for result in data["resultList"]["result"]]
    print(f"Found {len(pmcids)} PMCIDs. Saving to output directory.")
    _t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    for id in pmcids:
        response = requests.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{id}/fullTextXML")
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "lxml-xml")
            title = soup.find("article-title")
            abstract = soup.find("abstract")
            body = soup.find("body")
            text = ""
            if title:
                text += title.get_text() + "\n"
            if abstract:
                text += abstract.get_text() + "\n"
            if body:
                text += body.get_text()
            with open(os.path.join(output_dir, f"{id}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
    print(f"Saved {len(pmcids)} articles to {output_dir} in {time.time() - _t_start:.2f} seconds.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fetch Europe PMC articles.")
    parser.add_argument("--page_size", type=int, default=1000, help="Number of results to fetch")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save article texts")
    args = parser.parse_args()
    main(args.page_size, args.output_dir)
