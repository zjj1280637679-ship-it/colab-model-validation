import ast
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]

class PackageTests(unittest.TestCase):
    def test_python_syntax(self):
        for path in (ROOT / 'scripts').glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))

    def test_notebooks_clear_and_valid(self):
        for path in (ROOT / 'notebooks').glob('*.ipynb'):
            data = json.loads(path.read_text())
            self.assertEqual(data['nbformat'], 4)
            for cell in data['cells']:
                if cell['cell_type'] == 'code':
                    self.assertIsNone(cell['execution_count'])
                    self.assertEqual(cell['outputs'], [])
                    ast.parse(''.join(line for line in cell['source'] if not line.startswith('%')))

    def test_evidence_claims_have_files(self):
        data = json.loads((ROOT / 'evidence/manifest.json').read_text())
        for model in data['models']:
            if 'completed' in model['status'] or model['status'] == 'passed':
                self.assertTrue(model['evidence_files'])
            for file in model['evidence_files']:
                self.assertTrue((ROOT / 'evidence' / file).is_file())

    def test_audit_and_inference_evidence_consistency(self):
        audit=json.loads((ROOT/'evidence/qwen_checkpoint_audit.json').read_text())
        run=json.loads((ROOT/'evidence/qwen3_8_bf16.json').read_text())
        self.assertEqual(audit['model_only'], {})
        self.assertEqual(audit['shape_mismatch'], {})
        self.assertEqual(len(audit['checkpoint_only']),15)
        self.assertTrue(all(k.startswith('mtp.') for k in audit['checkpoint_only']))
        self.assertEqual(audit['checkpoint_total']-sum(run['dtype_counts'].values()),audit['checkpoint_only_total'])
        self.assertEqual(run['tests'][1]['answer'],'391')

    def test_notebook_lifecycle_and_embedded_sources(self):
        for model in ('qwen3_8','qwen_image_2_1'):
            n=json.loads((ROOT/'notebooks'/f'{model}_bf16.ipynb').read_text())
            code=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
            self.assertEqual(sum(c.count('arm_shutdown(shutdown_state, runtime.unassign)') for c in code),1)
            self.assertEqual(code[-1],'shutdown_now(shutdown_state, runtime.unassign)\n')
            self.assertNotIn('gpu_shutdown_timer',''.join(code))
            for source in ('preflight.py',model+'_bf16.py','export_evidence.py'):
                self.assertIn((ROOT/'scripts'/source).read_text(),code)

    def test_image_source_verified_before_import(self):
        source=(ROOT/'scripts/qwen_image_2_1_bf16.py').read_text()
        self.assertLess(source.index('for name in sys.modules'),source.index('subprocess.check_call'))
        self.assertLess(source.index('installed_diffusers = observed_distribution'),source.index('from diffusers import'))
        self.assertIn('"diffusers_commit":installed_diffusers["source"]["commit"]',source)
        self.assertIn('"diffusers_expected_commit":DIFFUSERS_COMMIT',source)

    def test_no_private_links(self):
        for directory in ('notebooks', 'scripts', 'evidence'):
            for path in (ROOT / directory).rglob('*'):
                if path.is_file() and '__pycache__' not in str(path):
                    content = path.read_text()
                    self.assertNotIn('colab.research.google.com/drive/', content)
                    self.assertNotIn('accounts.google.com/', content)

if __name__ == '__main__': unittest.main()
