"""
Enhanced feature extraction module for speech emotion recognition.
Extracts comprehensive audio features with robust error handling.
"""

import numpy as np
import librosa
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """Enhanced audio feature extractor with multiple feature types"""
    
    def __init__(self, sample_rate=22050, n_mfcc=13, n_chroma=12, n_mel=128):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_mel = n_mel
        self.scaler = StandardScaler()
        
    def load_audio(self, file_path, duration=30):
        """Load and preprocess audio file"""
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            
            # Remove silence
            y, _ = librosa.effects.trim(y, top_db=20)
            
            # Normalize
            y = librosa.util.normalize(y)
            
            if len(y) == 0:
                raise ValueError("Audio file is empty or too short")
                
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None, None
    
    def extract_mfcc_features(self, y, sr):
        """Extract MFCC features with statistical measures"""
        try:
            # Basic MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # Delta MFCC (velocity)
            mfcc_delta = librosa.feature.delta(mfcc)
            
            # Delta-Delta MFCC (acceleration)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            features = []
            
            # Statistical measures for each MFCC coefficient
            for mfcc_coeff in [mfcc, mfcc_delta, mfcc_delta2]:
                features.extend(np.mean(mfcc_coeff, axis=1))    # Mean
                features.extend(np.std(mfcc_coeff, axis=1))     # Standard deviation
                features.extend(np.max(mfcc_coeff, axis=1))     # Maximum
                features.extend(np.min(mfcc_coeff, axis=1))     # Minimum
                features.extend(np.median(mfcc_coeff, axis=1))  # Median
                features.extend(stats.skew(mfcc_coeff, axis=1)) # Skewness
                features.extend(stats.kurtosis(mfcc_coeff, axis=1)) # Kurtosis
            
            return features
            
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            return [0.0] * (self.n_mfcc * 3 * 7)  # 3 types × 7 statistics
    
    def extract_spectral_features(self, y, sr):
        """Extract spectral features"""
        try:
            features = []
            
            # Spectral centroid (brightness)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.extend([
                np.mean(spec_cent), np.std(spec_cent), 
                np.max(spec_cent), np.min(spec_cent)
            ])
            
            # Spectral bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features.extend([
                np.mean(spec_bw), np.std(spec_bw)
            ])
            
            # Spectral rolloff
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.extend([
                np.mean(spec_rolloff), np.std(spec_rolloff)
            ])
            
            # Spectral contrast
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features.extend([
                np.mean(spec_contrast), np.std(spec_contrast)
            ])
            
            # Spectral flatness
            spec_flatness = librosa.feature.spectral_flatness(y=y)
            features.extend([
                np.mean(spec_flatness), np.std(spec_flatness)
            ])
            
            return features
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
            return [0.0] * 12
    
    def extract_harmonic_features(self, y, sr):
        """Extract harmonic and percussive features"""
        try:
            features = []
            
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Harmonic features
            harmonic_energy = np.sum(y_harmonic ** 2)
            percussive_energy = np.sum(y_percussive ** 2)
            
            features.extend([
                harmonic_energy,
                percussive_energy,
                harmonic_energy / (percussive_energy + 1e-8)  # Harmonic-to-percussive ratio
            ])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=self.n_chroma)
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))
            
            return features
            
        except Exception as e:
            logger.warning(f"Harmonic feature extraction failed: {e}")
            return [0.0] * (3 + self.n_chroma * 2)
    
    def extract_rhythmic_features(self, y, sr):
        """Extract rhythm and tempo features"""
        try:
            features = []
            
            # Tempo and beats
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
            
            # Beat timing features
            if len(beats) > 1:
                beat_times = librosa.frames_to_time(beats, sr=sr)
                beat_intervals = np.diff(beat_times)
                
                features.extend([
                    np.mean(beat_intervals),
                    np.std(beat_intervals),
                    np.var(beat_intervals)
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Onset features
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            features.append(len(onset_times))  # Number of onsets
            
            if len(onset_times) > 1:
                onset_intervals = np.diff(onset_times)
                features.extend([
                    np.mean(onset_intervals),
                    np.std(onset_intervals)
                ])
            else:
                features.extend([0.0, 0.0])
            
            return features
            
        except Exception as e:
            logger.warning(f"Rhythmic feature extraction failed: {e}")
            return [0.0] * 7
    
    def extract_mel_features(self, y, sr):
        """Extract mel-spectrogram features"""
        try:
            # Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mel)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            features = []
            
            # Statistical measures of mel-spectrogram
            features.extend([
                np.mean(mel_spec_db),
                np.std(mel_spec_db),
                np.max(mel_spec_db),
                np.min(mel_spec_db),
                np.median(mel_spec_db)
            ])
            
            # Mel-frequency band energies (reduced for efficiency)
            mel_bands = np.mean(mel_spec_db, axis=1)
            features.extend(mel_bands[:20])  # First 20 mel bands
            
            return features
            
        except Exception as e:
            logger.warning(f"Mel feature extraction failed: {e}")
            return [0.0] * 25
    
    def extract_energy_features(self, y, sr):
        """Extract energy-based features"""
        try:
            features = []
            
            # RMS Energy
            rms = librosa.feature.rms(y=y)
            features.extend([
                np.mean(rms),
                np.std(rms),
                np.max(rms),
                np.min(rms)
            ])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features.extend([
                np.mean(zcr),
                np.std(zcr)
            ])
            
            # Energy entropy (frame-based energy distribution)
            frame_length = 2048
            hop_length = 512
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
            frame_energies = np.sum(frames ** 2, axis=0)
            
            if len(frame_energies) > 0:
                # Normalize energies
                frame_energies = frame_energies / (np.sum(frame_energies) + 1e-8)
                # Calculate entropy
                entropy = -np.sum(frame_energies * np.log2(frame_energies + 1e-8))
                features.append(entropy)
            else:
                features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.warning(f"Energy feature extraction failed: {e}")
            return [0.0] * 7
    
    def extract_pitch_features(self, y, sr):
        """Extract pitch-related features"""
        try:
            features = []
            
            # Fundamental frequency (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                       fmin=librosa.note_to_hz('C2'), 
                                                       fmax=librosa.note_to_hz('C7'))
            
            # Remove NaN values
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                features.extend([
                    np.mean(f0_clean),
                    np.std(f0_clean),
                    np.max(f0_clean),
                    np.min(f0_clean),
                    np.median(f0_clean)
                ])
                
                # Pitch variation features
                f0_diff = np.diff(f0_clean)
                features.extend([
                    np.mean(np.abs(f0_diff)),  # Mean absolute pitch change
                    np.std(f0_diff)            # Pitch change variability
                ])
            else:
                features.extend([0.0] * 7)
            
            # Voicing probability
            features.append(np.mean(voiced_probs))
            
            return features
            
        except Exception as e:
            logger.warning(f"Pitch feature extraction failed: {e}")
            return [0.0] * 8
    
    def extract_all_features(self, file_path):
        """Extract all features from audio file"""
        try:
            # Load audio
            y, sr = self.load_audio(file_path)
            if y is None:
                return None
            
            all_features = []
            
            # Extract different feature types
            mfcc_features = self.extract_mfcc_features(y, sr)
            spectral_features = self.extract_spectral_features(y, sr)
            harmonic_features = self.extract_harmonic_features(y, sr)
            rhythmic_features = self.extract_rhythmic_features(y, sr)
            mel_features = self.extract_mel_features(y, sr)
            energy_features = self.extract_energy_features(y, sr)
            pitch_features = self.extract_pitch_features(y, sr)
            
            # Combine all features
            all_features.extend(mfcc_features)
            all_features.extend(spectral_features)
            all_features.extend(harmonic_features)
            all_features.extend(rhythmic_features)
            all_features.extend(mel_features)
            all_features.extend(energy_features)
            all_features.extend(pitch_features)
            
            # Handle any remaining NaN or infinite values
            all_features = np.array(all_features)
            all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"Extracted {len(all_features)} features from {file_path}")
            return all_features.tolist()
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {file_path}: {e}")
            return None
    
    def fit_scaler(self, feature_list):
        """Fit the scaler on a list of feature vectors"""
        try:
            if feature_list and len(feature_list) > 0:
                self.scaler.fit(feature_list)
                logger.info("Scaler fitted successfully")
            else:
                logger.warning("Empty feature list provided for scaler fitting")
        except Exception as e:
            logger.error(f"Error fitting scaler: {e}")
    
    def transform_features(self, features):
        """Transform features using fitted scaler"""
        try:
            if features is not None:
                features_array = np.array(features).reshape(1, -1)
                return self.scaler.transform(features_array)[0].tolist()
            return None
        except Exception as e:
            logger.error(f"Error transforming features: {e}")
            return features
    
    def get_feature_names(self):
        """Return feature names for interpretability"""
        feature_names = []
        
        # MFCC feature names
        mfcc_stats = ['mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis']
        mfcc_types = ['mfcc', 'mfcc_delta', 'mfcc_delta2']
        for mfcc_type in mfcc_types:
            for i in range(self.n_mfcc):
                for stat in mfcc_stats:
                    feature_names.append(f"{mfcc_type}_{i}_{stat}")
        
        # Spectral feature names
        spectral_features = [
            'spec_cent_mean', 'spec_cent_std', 'spec_cent_max', 'spec_cent_min',
            'spec_bw_mean', 'spec_bw_std', 'spec_rolloff_mean', 'spec_rolloff_std',
            'spec_contrast_mean', 'spec_contrast_std', 'spec_flatness_mean', 'spec_flatness_std'
        ]
        feature_names.extend(spectral_features)
        
        # Harmonic feature names
        harmonic_features = ['harmonic_energy', 'percussive_energy', 'hp_ratio']
        for i in range(self.n_chroma):
            harmonic_features.extend([f'chroma_{i}_mean', f'chroma_{i}_std'])
        feature_names.extend(harmonic_features)
        
        # Rhythmic feature names
        rhythmic_features = [
            'tempo', 'beat_interval_mean', 'beat_interval_std', 'beat_interval_var',
            'onset_count', 'onset_interval_mean', 'onset_interval_std'
        ]
        feature_names.extend(rhythmic_features)
        
        # Mel feature names
        mel_features = ['mel_mean', 'mel_std', 'mel_max', 'mel_min', 'mel_median']
        mel_features.extend([f'mel_band_{i}' for i in range(20)])
        feature_names.extend(mel_features)
        
        # Energy feature names
        energy_features = [
            'rms_mean', 'rms_std', 'rms_max', 'rms_min',
            'zcr_mean', 'zcr_std', 'energy_entropy'
        ]
        feature_names.extend(energy_features)
        
        # Pitch feature names
        pitch_features = [
            'f0_mean', 'f0_std', 'f0_max', 'f0_min', 'f0_median',
            'pitch_change_mean', 'pitch_change_std', 'voicing_prob'
        ]
        feature_names.extend(pitch_features)
        
        return feature_names

# Usage example
if __name__ == "__main__":
    # Initialize extractor
    extractor = AudioFeatureExtractor()
    
    # Extract features from a single file
    features = extractor.extract_all_features("example_audio.wav")
    
    if features:
        print(f"Extracted {len(features)} features")
        
        # Get feature names
        feature_names = extractor.get_feature_names()
        print(f"Feature names: {len(feature_names)} total")
        
        # Example of processing multiple files
        file_list = ["audio1.wav", "audio2.wav", "audio3.wav"]
        all_features = []
        
        for file_path in file_list:
            file_features = extractor.extract_all_features(file_path)
            if file_features:
                all_features.append(file_features)
        
        if all_features:
            # Fit scaler
            extractor.fit_scaler(all_features)
            
            # Transform features
            normalized_features = [extractor.transform_features(f) for f in all_features]
            print(f"Processed {len(normalized_features)} audio files")
