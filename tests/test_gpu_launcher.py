"""Exercise run.sh with a fake Ollama server; never load a model or use a GPU."""
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import unittest
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GPULauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copyfile(ROOT / "run.sh", self.root / "run.sh")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.gpu = self.port - 11500
        self.env = {**os.environ, "FAKE_GPU": str(self.gpu),
                    "NO_PROXY": "upper.example", "no_proxy": "lower.example"}
        self.env["PATH"] = str(self.root / "bin") + os.pathsep + self.env["PATH"]
        scripts = {
            "bin/nvidia-smi": "import os; print(os.environ['FAKE_GPU'] + ', GPU-test')",
            ".venv/bin/python": "import os,sys,json; print(json.dumps({'args':sys.argv[1:],'NO_PROXY':os.environ.get('NO_PROXY'),'no_proxy':os.environ.get('no_proxy')}))",
            "bin/ollama": """import os,sys,json
from http.server import HTTPServer, BaseHTTPRequestHandler
if os.environ.get('FAKE_FAIL'):
    sys.exit(1)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data=b'{"version":"test"}'
        self.send_response(200)
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self,*args): pass
host,port=os.environ['OLLAMA_HOST'].split(':')
HTTPServer((host,int(port)),Handler).serve_forever()
""",
        }
        for name, source in scripts.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/usr/bin/python3\n" + source + "\n")
            path.chmod(0o755)

    def tearDown(self):
        for path in self.root.glob("outputs/ollama/*.pid"):
            pid = int(path.read_text())
            # Only stop this fixture's own server, never an arbitrary PID.
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
                if str(self.root / "bin/ollama").encode() in cmdline:
                    os.kill(pid, signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError):
                pass
        self.tmp.cleanup()

    def run_script(self, *args, env=None):
        return subprocess.run(["bash", str(self.root / "run.sh"), *args],
                              env=env or self.env, text=True, capture_output=True, timeout=15)

    def test_start_reuse_and_forward_arguments(self):
        args = ["--gpu", str(self.gpu), "--agent", "spring", "--episodes", "1"]
        first = self.run_script(*args)
        self.assertEqual(first.returncode, 0, first.stderr)
        pidfile = self.root / f"outputs/ollama/gpu_{self.gpu}.pid"
        pid = pidfile.read_text()
        result = json.loads(first.stdout)
        self.assertIn(f"client.base_url=http://127.0.0.1:{self.port}/v1", result["args"])
        self.assertNotIn("--gpu", result["args"])
        self.assertIn("spring", result["args"])
        self.assertEqual(result["NO_PROXY"], result["no_proxy"])
        self.assertTrue({"127.0.0.1", "localhost", "::1", "upper.example", "lower.example"}.issubset(result["NO_PROXY"].split(',')))
        second = self.run_script(*args)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(pidfile.read_text(), pid)
        self.assertIn("Reusing Ollama", second.stderr)

    def test_concurrent_launches_share_one_service(self):
        cmd = ["bash", str(self.root / "run.sh"), "--gpu", str(self.gpu)]
        processes = [subprocess.Popen(cmd, env=self.env, text=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE) for _ in range(2)]
        outputs = [process.communicate(timeout=15) for process in processes]
        self.assertTrue(all(process.returncode == 0 for process in processes), outputs)
        stderr = ''.join(err for _, err in outputs)
        self.assertEqual(stderr.count("Started Ollama"), 1)
        self.assertEqual(stderr.count("Reusing Ollama"), 1)

    def test_dry_run_help_and_default_do_not_start_services(self):
        for args in (["--gpu", str(self.gpu), "--dry-run"],
                     ["--gpu", str(self.gpu), "--help"], ["--agent", "react"]):
            with self.subTest(args=args):
                result = self.run_script(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse((self.root / "outputs").exists())
        self.assertNotIn("client.base_url=", result.stdout)

    def test_bad_gpu_or_conflicting_endpoint_fails(self):
        for args in (["--gpu"], ["--gpu", "-1"], ["--gpu", "missing"],
                     ["--gpu", "0"], ["--gpu", str(self.gpu), "--gpu", str(self.gpu)],
                     ["--gpu", str(self.gpu), "client.base_url=http://example.test/v1"]):
            with self.subTest(args=args):
                result = self.run_script(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse((self.root / "outputs").exists())

    def test_unmanaged_listener_is_not_reused(self):
        server = HTTPServer(("127.0.0.1", self.port), BaseHTTPRequestHandler)
        try:
            result = self.run_script("--gpu", str(self.gpu))
            self.assertEqual(result.returncode, 1)
            self.assertIn("unverified service", result.stderr)
            self.assertFalse(list(self.root.glob("outputs/ollama/*.pid")))
        finally:
            server.server_close()

    def test_failed_server_does_not_start_experiment(self):
        result = self.run_script("--gpu", str(self.gpu), env={**self.env, "FAKE_FAIL": "1"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("startup failed", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse(list(self.root.glob("outputs/ollama/*.pid")))


if __name__ == "__main__":
    unittest.main()
