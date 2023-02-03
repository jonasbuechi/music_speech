import numpy as np
from scipy.signal import hann
from scipy.io import wavfile
	
def get_mag_phs(re, im):
	return np.sqrt(re**2 + im**2), np.angle(re + im * 1j)

def get_r_i(mag, phs):
	return mag * np.cos(phs), mag * np.sin(phs)


def load_wav(filename):
	_, w = wavfile.read(filename)
	w = w.astype(np.float32)
	norm = np.max(np.abs(w))
	w = w / norm
	w -= np.mean(w)
	return w
	

def dft(signal, step_size=256, fft_size=512):
	n_steps = len(signal) // step_size
	s = []
	hann_win = hann(fft_size)
	for hop_i in range(n_steps):
		frame = signal[(hop_i * step_size):(hop_i * step_size + fft_size)]
		frame = np.pad(frame, (0, fft_size - len(frame)), 'constant')
		frame *= hann_win
		s.append(frame)
	s = np.array(s)
	N = s.shape[-1]
	k = np.reshape(np.linspace(0.0, 2 * np.pi / N * (N // 2), N // 2), [1, N // 2])
	x = np.reshape(np.linspace(0.0, N - 1, N), [N, 1])
	freqs = np.dot(x, k)
	reals = np.dot(s, np.cos(freqs)) * (2.0 / N)
	imags = np.dot(s, np.sin(freqs)) * (2.0 / N)
	return reals, imags


def idft(re, im, step_size=256, fft_size=512):
	N = re.shape[1] * 2
	k = np.reshape(np.linspace(0.0, 2 * np.pi / N * (N // 2), N // 2), [N // 2, 1])
	x = np.reshape(np.linspace(0.0, N - 1, N), [1, N])
	freqs = np.dot(k, x)
	signal = np.zeros((re.shape[0] * step_size + fft_size,))
	recon = np.dot(re, np.cos(freqs)) + np.dot(im, np.sin(freqs))
	for hop_i, frame in enumerate(recon):
		signal[(hop_i * step_size): (hop_i * step_size + fft_size)] += frame
	return signal
