"""
Audio Processing Module
Utilities for audio analysis and synthesis
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """
    Analyzes audio data for quality assessment
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize AudioAnalyzer
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
    
    @staticmethod
    def calculate_rms_energy(audio_data: np.ndarray) -> float:
        """
        Calculate RMS (Root Mean Square) energy
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            RMS energy value
        """
        if len(audio_data) == 0:
            return 0.0
        
        return np.sqrt(np.mean(audio_data ** 2))
    
    @staticmethod
    def calculate_zero_crossing_rate(audio_data: np.ndarray) -> float:
        """
        Calculate zero crossing rate
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Zero crossing rate
        """
        if len(audio_data) < 2:
            return 0.0
        
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / 2
        return zero_crossings / len(audio_data)
    
    def detect_silence(self, audio_data: np.ndarray, 
                      threshold: float = -40) -> Tuple[bool, float]:
        """
        Detect if audio contains significant silence
        
        Args:
            audio_data: Audio samples
            threshold: dB threshold for silence
            
        Returns:
            Tuple of (is_silent, silence_percentage)
        """
        rms = self.calculate_rms_energy(audio_data)
        
        # Convert to dB
        db = 20 * np.log10(rms + 1e-10)
        
        silence_ratio = 1.0 if db < threshold else 0.0
        
        return db < threshold, silence_ratio * 100
    
    def analyze_audio_quality(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive audio quality analysis
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Dictionary with quality metrics
        """
        return {
            'rms_energy': float(self.calculate_rms_energy(audio_data)),
            'zero_crossing_rate': float(self.calculate_zero_crossing_rate(audio_data)),
            'is_silent': self.detect_silence(audio_data)[0],
            'peak_amplitude': float(np.max(np.abs(audio_data))),
            'mean_amplitude': float(np.mean(np.abs(audio_data)))
        }


class WaveformGenerator:
    """
    Generates various waveforms for testing
    """
    
    @staticmethod
    def generate_sine_wave(frequency: float, duration: float, 
                          sample_rate: int = 44100, amplitude: float = 1.0) -> np.ndarray:
        """
        Generate sine wave
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Wave amplitude
            
        Returns:
            Audio samples as numpy array
        """
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        return amplitude * np.sin(2 * np.pi * frequency * t)
    
    @staticmethod
    def generate_white_noise(duration: float, sample_rate: int = 44100,
                            amplitude: float = 0.1) -> np.ndarray:
        """
        Generate white noise
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Noise amplitude
            
        Returns:
            Audio samples as numpy array
        """
        num_samples = int(sample_rate *duration)
        return amplitude * np.random.normal(0, 1, num_samples)
    
    @staticmethod
    def generate_square_wave(frequency: float, duration: float,
                            sample_rate: int = 44100, amplitude: float = 1.0) -> np.ndarray:
        """
        Generate square wave
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Wave amplitude
            
        Returns:
            Audio samples as numpy array
        """
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        return wave


__all__ = ['AudioAnalyzer', 'WaveformGenerator']
