"""Trust-boundary and reconstruction regressions for the public evidence package."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('scheduled_fallback_verify', ROOT/'verify.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class ReleaseVerification(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'evidence'
        shutil.copytree(ROOT,self.root,copy_function=os.link,
                        ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))

    def replace(self,name,data):
        path=self.root/name
        if path.exists():
            path.unlink()  # Break the private test hard link before writing.
        path.write_bytes(data)

    def rebind(self,name,data,arrays=None):
        manifest=json.loads((self.root/'manifest.json').read_text())
        entry=manifest['artifacts'][name]
        entry.update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        if arrays is not None:
            entry['native_sha256']=entry['sha256']
            entry['arrays']={k:{'dtype':str(v.dtype),'shape':list(v.shape)} for k,v in arrays.items()}
        self.replace(name,data)
        self.replace('manifest.json',json.dumps(manifest).encode())

    def test_complete_package_reconstructs_all_trials_and_pairs(self):
        report=verify.verify(self.root)
        self.assertEqual(report['failures']['trials'],24)
        self.assertEqual(len(report['paired_descriptive_outcomes']),12)
        self.assertFalse(report['safety_claim_supported'])

    def test_changed_raw_byte_is_rejected(self):
        name='raw/R1-nominal-continue_P.npz'
        data=bytearray((self.root/name).read_bytes());data[-1]^=1
        self.replace(name,bytes(data))
        with self.assertRaisesRegex(ValueError,'digest or size'):
            verify.verify(self.root)

    def test_extra_numerical_file_is_rejected(self):
        (self.root/'raw/extra.npz').write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'Unlisted'):
            verify.verify(self.root)

    def test_missing_trial_is_rejected_even_with_complete_files(self):
        manifest=json.loads((self.root/'manifest.json').read_text());manifest['trials'].pop()
        self.replace('manifest.json',json.dumps(manifest).encode())
        with self.assertRaisesRegex(ValueError,'Trial inventory'):
            verify.verify(self.root)

    def test_rehashed_wrong_modal_shape_is_rejected(self):
        name='raw/R1-nominal-continue_P.npz'
        with np.load(self.root/name,allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        arrays['command']=arrays['command'][:,:-1]
        buffer=io.BytesIO();np.savez(buffer,**arrays)
        self.rebind(name,buffer.getvalue(),arrays)
        with self.assertRaisesRegex(ValueError,'modal shape'):
            verify.verify(self.root)

    def test_rehashed_int64_predictor_view_is_rejected(self):
        name='models.npz'
        with np.load(self.root/name,allow_pickle=False) as archive:
            coefficients=archive['predictor_coefficients']
        arrays={'predictor_coefficients':coefficients.view(np.int64)}
        self.assertEqual(arrays['predictor_coefficients'].tobytes(),coefficients.tobytes())
        buffer=io.BytesIO();np.savez(buffer,**arrays)
        self.rebind(name,buffer.getvalue(),arrays)
        with self.assertRaisesRegex(ValueError,'Selected predictor'):
            verify.verify(self.root)

    def test_rehashed_protocol_change_is_rejected(self):
        config=json.loads((self.root/'config.json').read_text())
        config['physical']['wavelength_m']=1e-6
        self.rebind('config.json',json.dumps(config).encode())
        with self.assertRaisesRegex(ValueError,'Physical protocol'):
            verify.verify(self.root)

    def test_rehashed_reported_mean_change_is_reconstructed_and_rejected(self):
        report=json.loads((self.root/'assessment.json').read_text())
        report['primary_regimes']['nominal']['mean']+=.01
        self.rebind('assessment.json',json.dumps(report).encode())
        with self.assertRaisesRegex(ValueError,'numerical value'):
            verify.verify(self.root)

    def test_symlinked_evidence_is_rejected(self):
        path=self.root/'models.npz';path.unlink();path.symlink_to(ROOT/'models.npz')
        with self.assertRaisesRegex(ValueError,'Symlink'):
            verify.verify(self.root)


class NumericalContracts(unittest.TestCase):
    def test_student_quantile_matches_independent_reference(self):
        self.assertAlmostEqual(verify.t5_quantile(),3.1633814497486084,places=10)

    def test_scalar_two_pixel_physical_oracle(self):
        phase=np.array([[[.2,-.2]],[[.1,-.1]]],dtype=np.float64)
        basis=np.array([[[1.,-1.]]]);mask=np.ones((1,2));applied=np.array([[.05],[.02]])
        result=verify.physical_metrics(phase,applied,basis,mask,750e-9)
        np.testing.assert_allclose(result['variance'],np.array([.15**2,.08**2]),rtol=0,atol=1e-15)
        np.testing.assert_allclose(result['strehl'],np.cos([.15,.08])**2,rtol=0,atol=1e-15)

    def test_missing_pair_never_produces_survivor_mean(self):
        summary=verify.primary_interval([1.,2.,None,4.,5.,6.],'study')
        self.assertFalse(summary['complete']);self.assertIsNone(summary['mean']);self.assertIsNone(summary['interval'])

    def test_parent_and_absolute_paths_rejected(self):
        for name in ['../escape','/tmp/escape','a/../b','a\\b']:
            with self.subTest(name=name),self.assertRaises(ValueError):
                verify.relative_name(name)

    def test_oversized_npy_header_rejected_before_array_materialization(self):
        import zipfile
        raw=io.BytesIO()
        np.lib.format.write_array_header_1_0(raw,{'descr':'<f8','fortran_order':False,'shape':(10**12,)})
        packed=io.BytesIO()
        with zipfile.ZipFile(packed,'w',compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('x.npy',raw.getvalue())
        with self.assertRaisesRegex(ValueError,'oversized decoded array'):
            verify.archive_bytes(packed.getvalue(),{'x':{'dtype':'float64','shape':[10**12]}})


if __name__=='__main__':
    unittest.main()
