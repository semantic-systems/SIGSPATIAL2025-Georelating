import os
import re
import requests

from typing import List, Optional, Dict
from xml.etree import ElementTree as ET
from pydantic import BaseModel
from tqdm import tqdm


# Define Pydantic model
class ToponymData(BaseModel):
    osm_id: str
    coordinates: Optional[dict]
    toponym: str


class ArticleData(BaseModel):
    article_id: str
    article_text: str
    toponyms: List[str]
    toponym_data: List[ToponymData]


def parse_and_filter_xml(data_directory: str, file_name: str) -> List[Dict]:
    def extract_text_with_links(paragraph):
        paragraph_text = ET.tostring(paragraph, encoding='unicode')
        paragraph_text = re.sub(r'<link [^>]*>(.*?)</link>', r'\1', paragraph_text)
        paragraph_text = re.sub(r'<[^>]+>', '', paragraph_text)
        return paragraph_text.strip()

    def extract_relation_osm_id(link):
        osm_ids = link.get("osm").split()
        link_types = link.get("type").split()
        for osm_id, link_type in zip(osm_ids, link_types):
            if link_type == "relation":
                return osm_id
        return None

    file_path = os.path.join(data_directory, file_name)
    tree = ET.parse(file_path)
    root = tree.getroot()
    filtered_articles = []

    for entity in root.findall(".//entity"):
        article_osm_id = entity.get("osm")
        if "relation" not in (entity.get("type") or ""):
            continue

        article_text = ""
        toponyms = []
        toponym_osm_ids = []

        for paragraph in entity.findall(".//p"):
            text = extract_text_with_links(paragraph)
            links = paragraph.findall(".//link")

            for link in links:
                toponym_name = link.text
                osm_id = extract_relation_osm_id(link)
                if osm_id and toponym_name:
                    toponym_osm_ids.append(osm_id)
                    toponyms.append(toponym_name)

            article_text += text + " "

        filtered_articles.append({
            "article_osm_id": article_osm_id,
            "article_text": article_text.strip(),
            "toponyms": toponyms,
            "toponym_osm_ids": toponym_osm_ids
        })

    return filtered_articles


def query_osm_coordinates(osm_ids: List[str]) -> List[Dict]:
    results = []
    overpass_url = "https://overpass-api.de/api/interpreter"
    for osm_id in osm_ids:
        query = f"""
        [out:json];
        relation({osm_id});
        out center;
        """
        response = requests.get(overpass_url, params={"data": query})
        if response.status_code == 200:
            data = response.json()
            elements = data.get("elements", [])
            for element in elements:
                if "center" in element:
                    results.append({
                        "osm_id": osm_id,
                        "coordinates": {
                            "latitude": element["center"]["lat"],
                            "longitude": element["center"]["lon"]
                        }
                    })
                break
        else:
            print(f"Failed to fetch coordinates for OSM ID {osm_id}")
    return results


def process_articles(data_directory: str, file_name: str) -> List[ArticleData]:
    articles = parse_and_filter_xml(data_directory, file_name)

    final_articles = []
    osm_query_cache = {}

    for article in tqdm(articles, desc="Processing Articles", unit="article"):
        osm_ids = article["toponym_osm_ids"]
        toponym_data = []

        for osm_id, name in zip(osm_ids, article["toponyms"]):
            if osm_id in osm_query_cache:
                mapped_data = osm_query_cache[osm_id]
            else:
                # Query OSM for this ID
                osm_data = query_osm_coordinates([osm_id])
                mapped_data = osm_data[0] if osm_data else None
                osm_query_cache[osm_id] = mapped_data

            if mapped_data:
                toponym_data.append(ToponymData(
                    osm_id=osm_id,
                    coordinates=mapped_data.get("coordinates"),
                    toponym=name
                ))
            else:
                break  # Skip this article if any OSM ID fails
        else:
            final_articles.append(ArticleData(
                article_id=article["article_osm_id"],
                article_text=article["article_text"],
                toponyms=article["toponyms"],
                toponym_data=toponym_data
            ))

    return final_articles
