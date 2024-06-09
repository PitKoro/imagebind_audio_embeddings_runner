from collections import deque
import requests
from runners.utils import Item

class SimpleRunner:
    def __init__(
        self, parser, sink, logger, seed_urls, rate=100, max_tries=5
    ):
        self._parser = parser
        self._sink = sink
        self._logger = logger
        self._seed_urls = seed_urls
        self._rate = rate
        self._max_tries = max_tries
        self._to_process = deque()
        self._seen = set()
        for url in seed_urls:
            self._to_process.append(Item(url))
            self._seen.add(url)

    def _filter(self, urls):
        to_return = []
        for url in urls:
            if url in self._seen:
                continue
            to_return.append(url)
        return to_return
    
    def _process(self, item):
        resp = requests.get(item.url)
        resp.raise_for_status()
        content = resp.content
        result, next_urls = self._parser.parse(content, item.url)
        return result, next_urls

    def _write(self, item, result, error):
        self._logger.info(f"Writing {item.url} result = {result}, error = {error}")
        if error is not None:
            self._sink.write({'error': str(error), 'result': None})
            return
        self._sink.write({'error': None, 'result': result, 'url': item.url})

    def run(self):
        while self._to_process:
            item = self._to_process.popleft()
            try:
                result, next_urls = self._process(item)
            except Exception as e:
                self._logger.exception('Failed to process:')
                item.tries += 1
                if item.tries > self._max_tries:
                    self._write(item, result, e)
                self._to_process.append(item)
                continue
            if result is not None:
                self._write(item, result, None)
                
            for elem in self._filter(next_urls):
                self._to_process.append(Item(elem))
                self._seen.add(elem)