import unittest
import context
import json
import preprocess
from unittest import mock

class TestPreprocess(unittest.TestCase):
    def setUpClass(cls):
        cls._preprocessor = preprocess.Audio_Processor('')

    def test_set_audio_dir(self):
        self._preprocessor.set_audio_dir('../../ESC-50/audio')
        self.assertEqual(self._preprocessor._audio_dir, '../../ESC-50/audio')

if __name__ == '__main__':
    preproc_test = unittest.TestLoader().loadTestsFromTestCase(TestPreprocess)
    unittest.TextTestRunner(verbosity=1).run(api_test)