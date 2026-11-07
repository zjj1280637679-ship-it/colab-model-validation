import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec=importlib.util.spec_from_file_location('helpers',Path(__file__).resolve().parents[1]/'scripts/runtime_helpers.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

class FakeTimer:
    def __init__(self, seconds, callback):self.seconds=seconds;self.cancelled=False;self.started=False
    def start(self):self.started=True
    def cancel(self):self.cancelled=True

class RuntimeTests(unittest.TestCase):
    def test_single_deadline_rerun_cancels_without_extension(self):
        state={}
        h.arm_shutdown(state,lambda:None,timer_factory=FakeTimer,clock=lambda:100)
        old=state['timer']
        h.arm_shutdown(state,lambda:None,timer_factory=FakeTimer,clock=lambda:200)
        self.assertTrue(old.cancelled);self.assertEqual(state['timer'].seconds,1400)
        h.shutdown_now(state,lambda:None);self.assertTrue(state['timer'].cancelled)

    def test_failed_shutdown_retains_fallback(self):
        state={'timer':FakeTimer(1,None)}
        def fail():raise RuntimeError('release failed')
        with self.assertRaises(RuntimeError):h.shutdown_now(state,fail)
        self.assertFalse(state['timer'].cancelled)

    def test_unique_dirs_and_exact_allowlist_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root=h.new_report_dir(td,'qwen3_8');other=h.new_report_dir(td,'qwen3_8')
            self.assertNotEqual(root,other)
            for n in h.MODEL_PROFILES['qwen3_8']['files']:(root/n).write_text('{}')
            (root/'private.txt').write_text('must never export')
            (root/'nested').mkdir();(root/'nested'/'secret.json').write_text('secret')
            for attempt in range(2):
                archive,manifest=h.export_allowlisted(root,'qwen3_8')
                with zipfile.ZipFile(archive) as z:
                    self.assertEqual(set(z.namelist()),set(manifest)|{'sha256_manifest.json'})
                    self.assertEqual(json.loads(z.read('sha256_manifest.json')),manifest)
                    for n,digest in manifest.items():self.assertEqual(hashlib.sha256(z.read(n)).hexdigest(),digest)

    def test_missing_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=h.new_report_dir(td,'qwen3_8')
            with self.assertRaises(ValueError):h.export_allowlisted(root,'qwen3_8')
            for n in h.MODEL_PROFILES['qwen3_8']['files']:(root/n).write_text('{}')
            (root/'preflight.json').unlink();(root/'preflight.json').symlink_to(root/'runtime_versions.json')
            with self.assertRaises(ValueError):h.export_allowlisted(root,'qwen3_8')

    def test_resource_limits(self):
        for model,config in h.MODEL_PROFILES.items():
            h.require_resources(model,config['ram_gib'],config['disk_gib'])
            with self.assertRaises(RuntimeError):h.require_resources(model,config['ram_gib']-1,config['disk_gib'])
            with self.assertRaises(RuntimeError):h.require_resources(model,config['ram_gib'],config['disk_gib']-1)

    def test_distribution_unknown_and_source_redaction(self):
        class Dist:
            version='1.2.3'
            def read_text(self,name):return json.dumps({'url':'https://private.example/secret','vcs_info':{'commit_id':'a'*40}})
        with patch.object(h.importlib.metadata,'distribution',return_value=Dist()):
            self.assertEqual(h.observed_distribution('diffusers'),{'version':'1.2.3','source':None})
        with patch.object(h.importlib.metadata,'distribution',side_effect=h.importlib.metadata.PackageNotFoundError):
            self.assertIsNone(h.observed_distribution('diffusers'))

    def test_verified_installed_commit_recorded(self):
        class Dist:
            version='0.42.0.dev0'
            def read_text(self,name):return json.dumps({'url':'https://github.com/huggingface/diffusers.git','vcs_info':{'commit_id':'a'*40}})
        with patch.object(h.importlib.metadata,'distribution',return_value=Dist()):
            self.assertEqual(h.observed_distribution('diffusers')['source']['commit'],'a'*40)

if __name__=='__main__':unittest.main()
