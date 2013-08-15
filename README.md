Continuous wavelet transforms in Python.

A Clean Python implementation of the wavelet analysis outlined in [Torrence
and Compo][TC_Home] (BAMS, 1998)

[TC_home]: http://paos.colorado.edu/research/wavelets/
[TC_98]: http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf

![random data](https://raw.github.com/aaren/wavelets/images/random_data.png)


### Usage ###

```python
from wavelets import WaveletAnalysis

# given a signal x(t)
x = np.random.randn(1000)
# and a sample spacing
dt = 0.1

wa = WaveletAnalysis(x, dt=dt)

# wavelet power spectrum
power = wa.wavelet_power

# scales 
scales = wa.scales

# associated time vector
t = wa.time

# reconstruction of the original data
rx = wa.reconstruction()
```

#### How would you plot this? ####

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
T, S = np.meshgrid(t, scales)
ax.contourf(T, S, power, 100)
ax.set_yscale('log')
fig.savefig('test_wavelet_power_spectrum.png')
```

See the [tests](./tests.py) for more plotting examples.

#### What wavelet functions can I use? ####

The default is to use the Morlet. The Ricker (aka Mexican hat, aka
Marr) is also available.

You can write your own wavelet functions, in either time or
frequency. Just follow the example of Morlet in the source.

You specify the function to use when starting the analysis:

```python
from wavelets import Ricker

wa = WaveletAnalysis(data=x, wavelet=Ricker(), dt=dt)
```

### Requirements ###

- Python 2.7
- Numpy (developed with 1.7.1)
- Scipy (developed with 0.12.0)

Scipy is only used for `signal.fftconvolve` and `optimize.fsolve`,
and could potentially be removed.


### Issues ###

- Can't accurately compute `C_d` for an arbitrary wavelet. The
  empirically derived values are hardcoded into the given mother
  wavelets.
