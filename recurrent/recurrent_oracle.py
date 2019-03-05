import torch
import random
import math

# generate a time series for the tutorial task
def generate_sine_cosine(T = 100, tau = None, phaseshift = 0.):
	# first, select a wavelength at random in the range [20, 50]
	if(tau is None):
		tau = random.randrange(20, 51)
	# generate the times at which we sample the waves
	times = torch.Tensor(list(range(0, T)))
	# generate the input signal, i.e. the cosine
	X = torch.cos((times / tau + phaseshift) * 2. * math.pi)
	# generate the output signal, i.e. the sine
	Y = torch.sin((times / tau + phaseshift) * 2. * math.pi)
	# Set the first quarter-wavelength to zero
	Y[:int(tau / 4.)] = 0
	# unsqueeze the second axes to have standard form data
	X = X.unsqueeze(1)
	Y = Y.unsqueeze(1)
	return X, Y, times

def generate_exercise_data(T = 400, tau = None, phaseshift = 0., num_switches = None, noise_amplitude = 0.1):
	# first, select a wavelength at random in the range [20, 50]
	if(tau is None):
		tau = random.randrange(20, 51)
	# Then select the number of switches in the range [T / 200, T / 50]
	if(num_switches is None):
		num_switches = random.randrange(int(T / 200.), int(T / 50.))
	# generate the times at which we sample the waves
	times = torch.Tensor(list(range(0, T)))
	# generate the input signal, i.e. the sine wave
	X = torch.cos((times / tau + phaseshift) * 2. * math.pi)
	# generate the corresponding square wave by taking the
	# sign function of X
	sq_wave = torch.sign(X)
	# generate the corresponding triangle wave by taking the
	# arcsine of X
	tr_wave = 2 / math.pi * torch.asin(X)
	# initialize Y as zero tensor
	Y = torch.zeros(T)
	# initialize the state as zero tensor
	state = torch.zeros(T)
	# iterate over all switch times
	ts = [0]
	if(num_switches > 0):
		switch_interval = float(T) / num_switches
		for n in range(1, num_switches+1):
			ts.append(random.randrange(int((n-1) * switch_interval), int(n * switch_interval)))
			if(n % 2 == 1):
				# if n is odd, use a triangle wave until the current switch point
				Y[ts[n-1]:ts[n]] = tr_wave[ts[n-1]:ts[n]]
				state[ts[n-1]:ts[n]] = 0.
			else:
				# otherwise use a square wave
				Y[ts[n-1]:ts[n]] = sq_wave[ts[n-1]:ts[n]]
				state[ts[n-1]:ts[n]] = 1.
	if(num_switches % 2 == 0):
		# if n is even, use a triangle wave for the remainder
		Y[ts[-1]:] = tr_wave[ts[-1]:]
		state[ts[-1]:] = 0.
	else:
		# otherwise use a square wave
		Y[ts[-1]:] = sq_wave[ts[-1]:]
		state[ts[-1]:] = 1.
	# add noise to X
	X += torch.randn(T) * noise_amplitude
	# unsqueeze the second axes to have standard form data
	X = X.unsqueeze(1)
	Y = Y.unsqueeze(1)
	state = state.unsqueeze(1)
	# generate the control signal and concatenate it with X
	X2 = torch.zeros(T, 1)
	X2[ts, :] = 1.
	X = torch.cat((X, X2), dim=1)
	return X, Y, times, state
