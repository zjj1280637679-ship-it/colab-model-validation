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
            if model['status'] in ('passed', 'inference_completed'):
                self.assertTrue(model['evidence_files'])
            for file in model['evidence_files']:
                self.assertTrue((ROOT / 'evidence' / file).is_file())

    def test_no_private_links(self):
        for directory in ('notebooks', 'scripts', 'evidence'):
            for path in (ROOT / directory).rglob('*'):
                if path.is_file() and '__pycache__' not in str(path):
                    content = path.read_text()
                    self.assertNotIn('colab.research.google.com/drive/', content)
                    self.assertNotIn('accounts.google.com/', content)

if __name__ == '__main__': unittest.main()
