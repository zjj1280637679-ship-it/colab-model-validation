import ast
import hashlib
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
H3=ROOT/'h3_author'

class H3PackagingTests(unittest.TestCase):
    def test_frozen_program_identity(self):
        m=json.loads((H3/'SHA256.json').read_text())
        self.assertEqual(m['program_source_sha256'],'9f6bb9f7189e124f3b908997faa7d9362686c75f2a3e9d117232490934cb7dc8')
        for name,digest in m['files'].items():self.assertEqual(hashlib.sha256((H3/name).read_bytes()).hexdigest(),digest)
        cell1=(H3/'cell_01.py').read_text();cell2=(H3/'cell_02.py').read_text()
        self.assertEqual(hashlib.sha256((cell1.split('\n',1)[1]+'\n'+cell2).encode()).hexdigest(),m['program_source_sha256'])
        evidence=json.loads((ROOT/'evidence/h3_v2_source_and_settings.json').read_text())
        self.assertEqual(hashlib.sha256(cell1.encode()).hexdigest(),evidence['frozen_cell_01_sha256'])
        self.assertEqual(hashlib.sha256(cell2.encode()).hexdigest(),evidence['cell_02_sha256'])

    def test_defaults_off_and_notebook_sources_match(self):
        values={}
        for n in ast.parse((H3/'config.py').read_text()).body:
            if isinstance(n,ast.Assign):values[n.targets[0].id]=ast.literal_eval(n.value)
        self.assertIs(values['ALLOW_THIS_RUN'],False)
        self.assertIs(values['I_READ_H3_LICENSE'],False)
        self.assertIs(values['I_UNDERSTAND_UNRESOLVED_LICENSE_RISK'],False)
        self.assertEqual(values['MODE'],'CPU_NO_WEIGHTS')
        self.assertEqual(values['KNOWN_APPLICABLE_BLOCK_STATUS'],'not_assessed')
        nb=json.loads((H3/'minimax_h3_author_bf16.ipynb').read_text())
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        for filename in ['config.py','cell_01.py','cell_02.py']:self.assertIn((H3/filename).read_text(),code)
        for c in nb['cells']:
            if c['cell_type']=='code':
                self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count']);ast.parse(''.join(c['source']))

    def test_only_two_workflow_asset_changes(self):
        original=json.loads((H3/'upstream/minimax_h3_t2v.json').read_text())
        actual=json.loads((H3/'minimax_h3_t2v.json').read_text())
        original['6']['inputs']['unet_name']='minimax_h3_fl2va_bf16.safetensors'
        original['13']['inputs']['clip_name']='qwen3vl_32b_minimax_h3_bf16.safetensors'
        self.assertEqual(original,actual);self.assertEqual(len(actual),14)

    def test_native_storage_evidence_and_media_limits(self):
        d=json.loads((ROOT/'evidence/h3_v2_gpu_summary.json').read_text())
        self.assertNotIn('installed_freeze',d);self.assertNotIn('outputs',d)
        self.assertEqual(set(d['execution_status']),{'status_str','completed'})
        self.assertTrue(d['execution_status']['completed'])
        self.assertEqual(d['gpu_peak_memory'],'NOT_MEASURED')
        self.assertEqual(d['compute_autocast_dtype'],'NOT_MEASURED')
        self.assertFalse(d['review_limits']['independent_media_hash_verified'])
        self.assertFalse(d['review_limits']['independent_listening_passed'])
        for row in d['precision']:self.assertEqual(set(row['parameter_numel_by_device']),{'cpu'})
        m=d['media'];self.assertEqual((m['width'],m['height'],m['frames'],m['fps'],m['sample_rate'],m['channels']),(864,480,124,24.0,32000,2))
        cpu=json.loads((ROOT/'evidence/h3_v2_cpu_summary.json').read_text())
        self.assertFalse(cpu['model_loaded']);self.assertFalse(cpu['inference_executed'])
        self.assertEqual(cpu['program_source_sha256'],d['program_source_sha256'])

    def test_license_and_python_syntax(self):
        self.assertEqual((H3/'LICENSE').read_bytes(),(H3/'upstream/LICENSE').read_bytes())
        self.assertIn('Copyright (c) 2026 Yuki Shindo',(H3/'LICENSE').read_text())
        self.assertIn('THE SOFTWARE IS PROVIDED "AS IS"',(H3/'LICENSE').read_text())
        for p in H3.glob('*.py'):ast.parse(p.read_text(),filename=str(p))

if __name__=='__main__':unittest.main()
