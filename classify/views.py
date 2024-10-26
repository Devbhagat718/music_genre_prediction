from django.shortcuts import render
from django.core.files.storage import default_storage
from django.http import JsonResponse
import joblib
import numpy as np
import librosa

# Load pre-trained model and transformers
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')
model = joblib.load('model.pkl')

def upload_audio(request):
    if request.method == 'POST' and request.FILES['audio_file']:
        # Save uploaded file
        audio_file = request.FILES['audio_file']
        path = default_storage.save('temp/' + audio_file.name, audio_file)

        # Extract features using librosa
        y, sr = librosa.load(path, duration=30)
        
        # Compute all the required features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
        chroma_stft_var = librosa.feature.chroma_stft(y=y, sr=sr).var()
        rms = librosa.feature.rms(y=y).mean()
        rms_var = librosa.feature.rms(y=y).var()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_centroid_var = librosa.feature.spectral_centroid(y=y, sr=sr).var()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_bandwidth_var = librosa.feature.spectral_bandwidth(y=y, sr=sr).var()
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        rolloff_var = librosa.feature.spectral_rolloff(y=y, sr=sr).var()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        zero_crossing_rate_var = librosa.feature.zero_crossing_rate(y).var()
        harmony = librosa.effects.harmonic(y).mean()
        harmony_var = librosa.effects.harmonic(y).var()
        perceptr = librosa.effects.percussive(y).mean()
        perceptr_var = librosa.effects.percussive(y).var()
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = [np.mean(mfcc) for mfcc in mfccs]
        mfcc_var = [np.var(mfcc) for mfcc in mfccs]

        # Compile features into an array
        features = [
            3.0,  # Dummy placeholder for length, replace with actual length if needed
            chroma_stft, chroma_stft_var, rms, rms_var, spectral_centroid, spectral_centroid_var,
            spectral_bandwidth, spectral_bandwidth_var, rolloff, rolloff_var,
            zero_crossing_rate, zero_crossing_rate_var, harmony, harmony_var, perceptr, perceptr_var,
            tempo, *mfcc_mean, *mfcc_var
        ]

        # Reshape features array to ensure it has the expected 58 features
        features = np.array(features).reshape(1, -1)

        # Scale, transform, and predict genre
        features_scaled = scaler.transform(features)
        features_poly = poly.transform(features_scaled)
        genre = model.predict(features_poly)[0]

        # Remove temporary file
        default_storage.delete(path)
        
        return JsonResponse({'genre': genre})

    return render(request, 'upload.html')
