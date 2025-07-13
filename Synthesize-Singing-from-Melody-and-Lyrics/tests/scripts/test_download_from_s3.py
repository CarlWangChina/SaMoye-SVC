"""Test cases for download form s3."""

import unittest
from pathlib import Path

from singer.configs import *
from singer.utils import logging
from singer.utils import download_from_index_music

logger = logging.get_logger(__name__)

class TestDownloadFromS3(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()

    def test_download_from_s3(self):
        logger.info('test_download_from_s3')

        hparams = get_hparams()

        logger.info(f"hparams = {hparams}")

        self.assertIsNotNone(logger)
        self.assertGreater(len(hparams), 0)

        download_from_index_music([2002434, 2002444, 2002565], rewrite=False)

if __name__ == '__main__':
    unittest.main()

