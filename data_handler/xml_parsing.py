from typing import List, Dict, Optional, Set, Union
import xml.etree.ElementTree as ET
import pandas as pd
import os


class XMLDataHandler:
    def __init__(self, data_directory: str) -> None:
        self.data_directory = data_directory
        self.articles_df: pd.DataFrame = pd.DataFrame()
        self.toponyms_with_gaztag_df: pd.DataFrame = pd.DataFrame()
        self.toponyms_without_gaztag_df: pd.DataFrame = pd.DataFrame()

    def parse_xml(self, file_name: str) -> None:
        file_path = os.path.join(self.data_directory, file_name)
        tree = ET.parse(file_path)
        root = tree.getroot()

        articles_data: List[Dict[str, Union[str, None]]] = []
        toponyms_with_gaztag_data: List[Dict[str, Union[str, None]]] = []
        toponyms_without_gaztag_data: List[Dict[str, Union[str, None]]] = []

        for article in root.findall('article'):
            article_data = self._extract_article_data(article)
            articles_data.append(article_data)

            for toponym in article.find('toponyms').findall('toponym'):
                toponym_data = self._extract_toponym_data(article, toponym)
                if toponym.find('gaztag') is not None:
                    toponyms_with_gaztag_data.append(toponym_data)
                else:
                    toponyms_without_gaztag_data.append(toponym_data)

        self.articles_df = pd.DataFrame(articles_data)
        self.toponyms_with_gaztag_df = pd.DataFrame(toponyms_with_gaztag_data)
        self.toponyms_without_gaztag_df = pd.DataFrame(toponyms_without_gaztag_data)

    def _extract_article_data(self, article: ET.Element) -> Dict[str, Union[str, None]]:
        return {
            'docid': article.get('docid'),
            'feedid': article.find('feedid').text,
            'title': article.find('title').text,
            'domain': article.find('domain').text,
            'url': article.find('url').text,
            'dltime': article.find('dltime').text,
            'text': article.find('text').text
        }

    def _extract_toponym_data(self, article: ET.Element, toponym: ET.Element) -> Dict[str, Union[str, None]]:
        toponym_data = {
            'docid': article.get('docid'),
            'start': toponym.find('start').text,
            'end': toponym.find('end').text,
            'phrase': toponym.find('phrase').text,
        }

        gaztag = toponym.find('gaztag')
        if gaztag is not None:
            toponym_data.update(self._extract_gaztag_data(gaztag))

        return toponym_data

    def _extract_gaztag_data(self, gaztag: ET.Element) -> Dict[str, Union[str, None]]:
        return {
            'geonameid': gaztag.get('geonameid'),
            'name': gaztag.find('name').text if gaztag.find('name') is not None else None,
            'fclass': gaztag.find('fclass').text if gaztag.find('fclass') is not None else None,
            'fcode': gaztag.find('fcode').text if gaztag.find('fcode') is not None else None,
            'lat': gaztag.find('lat').text if gaztag.find('lat') is not None else None,
            'lon': gaztag.find('lon').text if gaztag.find('lon') is not None else None,
            'country': gaztag.find('country').text if gaztag.find('country') is not None else None,
            'admin1': gaztag.find('admin1').text if gaztag.find('admin1') is not None else None
        }

    def get_all_articles(self) -> pd.DataFrame:
        return self.articles_df

    def get_article(self, docid: str) -> Optional[Dict[str, Union[str, None]]]:
        article = self.articles_df[self.articles_df['docid'] == docid]
        return article.to_dict(orient='records')[0] if not article.empty else None

    def get_all_articles_for_prompting(self) -> pd.DataFrame:
        return self.articles_df[['docid', 'title', 'text']]

    def get_article_for_prompting(self, docid: str) -> Optional[Dict[str, Union[str, None]]]:
        article = self.articles_df[self.articles_df['docid'] == docid][['docid', 'title', 'text']]
        return article.to_dict(orient='records')[0] if not article.empty else None

    def get_all_toponyms_with_gaztag(self) -> pd.DataFrame:
        return self.toponyms_with_gaztag_df

    def get_all_toponyms_without_gaztag(self) -> pd.DataFrame:
        return self.toponyms_without_gaztag_df

    def get_toponyms_for_article(self, docid: str) -> List[Dict[str, Union[str, None]]]:
        toponyms = self.toponyms_with_gaztag_df[self.toponyms_with_gaztag_df['docid'] == docid]
        return toponyms.to_dict(orient='records')

    def get_short_toponyms_for_article(self, docid: str) -> List[str]:
        phrases = self.toponyms_with_gaztag_df[self.toponyms_with_gaztag_df['docid'] == docid]['phrase']
        return phrases.tolist()

    def find_all_fields(self, file_name: str) -> Set[str]:
        file_path = os.path.join(self.data_directory, file_name)
        tree = ET.parse(file_path)
        root = tree.getroot()
        fields: Set[str] = set()

        def traverse(element: ET.Element) -> None:
            fields.add(element.tag)
            for child in element:
                traverse(child)

        traverse(root)
        return fields

    def count_duplicate_toponyms(self) -> int:
        if self.toponyms_with_gaztag_df.empty:
            return 0

        duplicate_counts = self.toponyms_with_gaztag_df.groupby('docid').apply(
            lambda x: x['phrase'].duplicated().sum()
        )
        total_duplicates = duplicate_counts.sum()

        return total_duplicates

    def get_random_articles_for_evaluation(self, seed: int, n: int = 100) -> pd.DataFrame:
        """
        Returns a sample of `n` articles using the given `seed` for reproducibility.

        :param seed: The seed for the random number generator.
        :param n: The number of articles to sample. Default is 100.
        :return: A tuple (sampled_articles, corresponding_toponyms).
        """
        # Ensure we have enough articles
        n = min(n, (len(self.articles_df)-10))

        # Sample articles reproducibly
        sampled_articles = self.articles_df[10:].sample(n=n, random_state=seed)

        return sampled_articles
