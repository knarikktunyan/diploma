from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from config import CLASS_NAMES, FEATURE_NAMES
from features import extract_features
from local_features import LOCAL_FEATURE_NAMES, LOCAL_FEATURE_VERSION, extract_local_features
from local_feature_experiment import feature_names, load_experimental_model, predict_experimental_signal


class LocalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.time=np.arange(500)/100000
        self.signal=np.sin(2*np.pi*1000*self.time+0.2)

    def test_features_finite_and_preserve_signal(self):
        original=self.signal.copy()
        features=extract_local_features(self.signal)
        self.assertEqual(list(features),LOCAL_FEATURE_NAMES)
        self.assertTrue(all(np.isfinite(value) for value in features.values()))
        np.testing.assert_array_equal(self.signal,original)
        self.assertLess(features['crossing_period_variation'],0.01)

    def test_changes_raise_expected_local_measurements(self):
        normal=extract_local_features(self.signal)
        amplitude=self.signal.copy(); amplitude[150:300]*=2
        offset=self.signal.copy(); offset[250:]+=0.7
        changed=self.signal.copy()
        phase=2*np.pi*1000*self.time[200]+0.2
        changed[200:]=np.sin(phase+2*np.pi*1600*(self.time[200:]-self.time[200]))
        self.assertGreater(extract_local_features(amplitude)['window_rms_variation'],normal['window_rms_variation'])
        self.assertGreater(extract_local_features(offset)['window_mean_variation'],normal['window_mean_variation'])
        self.assertGreater(extract_local_features(changed)['crossing_period_variation'],normal['crossing_period_variation'])

    def test_constant_and_invalid_signals(self):
        self.assertEqual(extract_local_features(np.zeros(500)),dict.fromkeys(LOCAL_FEATURE_NAMES,0.0))
        self.assertTrue(all(np.isfinite(v) for v in extract_local_features(np.ones(500)).values()))
        for signal in [[],[1]*31,[1,np.nan]*250,[[1]*500]]:
            with self.assertRaises(ValueError): extract_local_features(signal)

    def test_candidate_saved_csv_prediction_and_schema_rejection(self):
        features={**extract_features(self.signal,100000),**extract_local_features(self.signal)}
        columns=feature_names('combined')
        model=RandomForestClassifier(n_estimators=5,random_state=42)
        rng=np.random.default_rng(42)
        signals=[self.signal, self.signal+rng.normal(0,0.25,500),
                 self.signal+0.2*np.sin(2*np.pi*2000*self.time),
                 self.signal*np.where(self.time>=0.0025,2,1)]
        training=[{**extract_features(signal,100000),**extract_local_features(signal)} for signal in signals]
        model.fit(pd.DataFrame(training,columns=columns),CLASS_NAMES)
        artifact=dict(model=model,features=columns,classes=CLASS_NAMES,feature_group='combined',
                      local_feature_version=LOCAL_FEATURE_VERSION,
                      preprocessing={'remove_dc':False,'filter_signal':False,'normalize':False})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pkl'; csv=Path(folder)/'signal.csv'
            joblib.dump(artifact,path)
            pd.DataFrame({'time':self.time,'amplitude':self.signal}).to_csv(csv,index=False)
            expected=model.predict(pd.DataFrame([features],columns=columns))[0]
            self.assertEqual(predict_experimental_signal(csv,path)['predicted_class'].lower(),expected)
            self.assertEqual(FEATURE_NAMES,columns[:10])
            artifact['features']=['filename']; joblib.dump(artifact,path)
            with self.assertRaisesRegex(ValueError,'schema'): load_experimental_model(path)


if __name__=='__main__': unittest.main()
