"""Download-boundary tests, without importing NLP models or datasets.

Mock tests compile only the helper definitions from the actual source file.
Loopback tests load that file with real download dependencies and stub only its
local logger/settings imports. Neither suite validates the full NLP runtime.
"""
import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock
import zipfile


class HTTPError(Exception):
    def __init__(self, message='', response=None):
        super().__init__(message)
        self.response = response


class InstallHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = self.tmp.name
        self.response = mock.MagicMock()
        self.response.__enter__.return_value = self.response
        self.response.status_code = 200
        self.response.headers = {'Content-Length': '3'}
        self.response.iter_content.return_value = [b'a', b'', b'bc']
        self.requests = types.SimpleNamespace(
            get=mock.Mock(return_value=self.response), HTTPError=HTTPError)
        self.progress = mock.Mock()
        self.tqdm = types.SimpleNamespace(
            tqdm=mock.Mock(return_value=self.progress))
        self.lock = mock.MagicMock()
        self.filelock = types.SimpleNamespace(
            FileLock=mock.Mock(return_value=self.lock))
        source = Path(os.environ.get(
            'TEXTFLINT_INSTALL_SOURCE',
            str(Path(__file__).resolve().parents[3] /
                'textflint/common/utils/install.py')))
        parsed = ast.parse(source.read_text(encoding='utf-8'))
        names = {'textflint_url', 'path_in_cache', 'unzip_file',
                 'download_if_needed', 'http_get'}
        parsed.body = [node for node in parsed.body
                       if isinstance(node, ast.FunctionDef) and node.name in names]
        self.helpers = dict(os=os, shutil=shutil, tempfile=tempfile,
                            zipfile=zipfile, filelock=self.filelock,
                            requests=self.requests, tqdm=self.tqdm,
                            logger=mock.Mock(), CACHE_DIR=self.cache)
        exec(compile(parsed, str(source), 'exec'), self.helpers)

    def test_https_url(self):
        self.assertEqual(
            self.helpers['textflint_url']('folder/file.zip'),
            'https://textflint.oss-cn-beijing.aliyuncs.com/download/folder/file.zip')

    def test_success_stream_and_transport_options(self):
        output = io.BytesIO()
        proxies = {'https': 'http://example.invalid:8080'}
        self.helpers['http_get']('fixture.bin', output, proxies=proxies)
        self.assertEqual(output.getvalue(), b'abc')
        self.requests.get.assert_called_once_with(
            self.helpers['textflint_url']('fixture.bin'), stream=True,
            proxies=proxies, timeout=(10, 60), allow_redirects=False)
        self.response.raise_for_status.assert_called_once_with()
        self.response.__exit__.assert_called_once()
        self.progress.close.assert_called_once_with()

    def test_http_errors_and_redirects_never_read_or_write_body(self):
        for status in (204, 206, 301, 302, 304, 307, 308, 400, 403, 404, 500):
            with self.subTest(status=status):
                self.response.reset_mock()
                self.response.status_code = status
                self.response.raise_for_status.side_effect = (
                    HTTPError('HTTP error', response=self.response)
                    if status >= 400 else None)
                output = io.BytesIO()
                with self.assertRaises(HTTPError):
                    self.helpers['http_get']('fixture.bin', output)
                self.assertEqual(output.getvalue(), b'')
                self.response.iter_content.assert_not_called()
                self.response.__exit__.assert_called_once()

    def test_stream_failure_closes_response_and_progress(self):
        def broken():
            yield b'a'
            raise OSError('interrupted')
        self.response.iter_content.return_value = broken()
        with self.assertRaises(OSError):
            self.helpers['http_get']('fixture.bin', io.BytesIO())
        self.response.__exit__.assert_called_once()
        self.progress.close.assert_called_once_with()

    def test_tls_failure_has_no_insecure_fallback(self):
        self.requests.get.side_effect = OSError('certificate validation failed')
        with self.assertRaises(OSError):
            self.helpers['http_get']('fixture.bin', io.BytesIO())
        self.requests.get.assert_called_once()
        self.assertTrue(self.requests.get.call_args[0][0].startswith('https://'))
        self.assertNotIn('verify', self.requests.get.call_args[1])

    def test_failure_cleans_temp_and_releases_lock_then_retry_succeeds(self):
        self.response.raise_for_status.side_effect = HTTPError('not found')
        with self.assertRaises(HTTPError):
            self.helpers['download_if_needed']('fixture.bin')
        self.assertEqual(os.listdir(self.cache), [])
        self.lock.__exit__.assert_called_once()
        self.response.raise_for_status.side_effect = None
        result = self.helpers['download_if_needed']('fixture.bin')
        self.assertEqual(Path(result).read_bytes(), b'abc')
        self.assertEqual(os.listdir(self.cache), ['fixture.bin'])
        self.assertEqual(self.lock.__exit__.call_count, 2)

    def test_connection_failure_leaves_no_cache_or_temp(self):
        self.requests.get.side_effect = OSError('connection failed')
        with self.assertRaises(OSError):
            self.helpers['download_if_needed']('fixture.bin')
        self.assertEqual(os.listdir(self.cache), [])
        self.lock.__exit__.assert_called_once()

    def test_stream_failure_leaves_no_cache_or_temp(self):
        def broken():
            yield b'partial'
            raise OSError('interrupted')
        self.response.iter_content.return_value = broken()
        with self.assertRaises(OSError):
            self.helpers['download_if_needed']('fixture.bin')
        self.assertEqual(os.listdir(self.cache), [])
        self.lock.__exit__.assert_called_once()
        self.response.__exit__.assert_called_once()
        self.progress.close.assert_called_once_with()

    def test_zip_success_preserves_exact_fixture_bytes(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as packed:
            packed.writestr('data.txt', b'exact fixture bytes')
        payload = archive.getvalue()
        self.response.iter_content.return_value = [payload[:20], payload[20:]]
        self.response.headers = {'Content-Length': str(len(payload))}
        result = self.helpers['download_if_needed']('fixture')
        self.assertEqual((Path(result) / 'data.txt').read_bytes(),
                         b'exact fixture bytes')
        self.assertEqual(os.listdir(self.cache), ['fixture'])
        self.lock.__exit__.assert_called_once()

    def test_cache_hit_performs_no_request(self):
        cached = Path(self.cache) / 'fixture.bin'
        cached.write_bytes(b'cached')
        result = self.helpers['download_if_needed']('fixture.bin')
        self.assertEqual(result, str(cached))
        self.requests.get.assert_not_called()
        self.lock.__exit__.assert_called_once()


class RealDependencyHTTPTests(unittest.TestCase):
    """Exercise actual Requests/FileLock/tqdm without loading the NLP package."""

    def setUp(self):
        try:
            import requests
            import filelock
            import tqdm
        except ImportError as error:
            self.skipTest('Download dependencies unavailable: {}'.format(error))
        self.requests = requests
        self.filelock = filelock
        self.tqdm = tqdm
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name)
        source = Path(os.environ.get(
            'TEXTFLINT_INSTALL_SOURCE',
            str(Path(__file__).resolve().parents[3] /
                'textflint/common/utils/install.py')))
        # Stub only TextFlint's two local imports, not its download dependencies.
        package = '_textflint_download_test.common.utils'
        logger = types.ModuleType(package + '.logger')
        logger.logger = logging.getLogger(__name__)
        settings = types.ModuleType('_textflint_download_test.common.settings')
        settings.CACHE_DIR = str(self.cache)
        modules = {logger.__name__: logger, settings.__name__: settings}
        with mock.patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(
                package + '.install', source)
            self.install = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.install)

        self.paths = []
        paths = self.paths

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                paths.append(self.path)
                status = int(self.path[1:]) if self.path[1:].isdigit() else 200
                if self.path == '/truncated':
                    self.send_response(200)
                    self.send_header('Content-Length', '100')
                    self.end_headers()
                    self.wfile.write(b'partial')
                    self.close_connection = True
                    return
                self.send_response(status)
                if 300 <= status < 400:
                    self.send_header('Location', '/target')
                self.send_header('Content-Length', '3')
                self.end_headers()
                self.wfile.write(b'abc')

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop_server():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.addCleanup(stop_server)
        # HTTP is confined to loopback; the separate live smoke uses real TLS.
        self.install.textflint_url = lambda name: 'http://127.0.0.1:{}/{}'.format(
            server.server_port, name)

    def assert_clean_and_unlocked(self, name):
        self.assertEqual(sorted(p.name for p in self.cache.iterdir()),
                         [name + '.lock'])
        with self.filelock.FileLock(str(self.cache / (name + '.lock')), timeout=0):
            pass

    def test_real_download_and_cache_hit(self):
        result = self.install.download_if_needed('fixture.bin')
        self.assertEqual(Path(result).read_bytes(), b'abc')
        self.assertEqual(self.install.download_if_needed('fixture.bin'), result)
        self.assertEqual(self.paths, ['/fixture.bin'])

    def test_real_status_errors_and_redirects_do_not_enter_cache(self):
        expected_locks = []
        for status in (204, 206, 301, 302, 303, 304, 307, 308, 403, 404, 500):
            with self.subTest(status=status):
                name = str(status)
                with self.assertRaises(self.requests.HTTPError):
                    self.install.download_if_needed(name)
                self.assertEqual(self.paths[-1], '/' + name)
                self.assertNotIn('/target', self.paths)
                expected_locks.append(name + '.lock')
                self.assertEqual(sorted(p.name for p in self.cache.iterdir()),
                                 sorted(expected_locks))
                with self.filelock.FileLock(
                        str(self.cache / (name + '.lock')), timeout=0):
                    pass

    def test_real_truncated_stream_cleans_temp_and_unlocks(self):
        with self.assertRaises(self.requests.exceptions.ChunkedEncodingError):
            self.install.download_if_needed('truncated')
        self.assert_clean_and_unlocked('truncated')


if __name__ == '__main__':
    unittest.main()
