Continuous wavelet transforms in Python.
----------------------------------------

<a href='https://travis-ci.org/aaren/wavelets'>
<img src='https://secure.travis-ci.org/aaren/wavelets.png?branch=master'></a>

A Clean Python implementation of the wavelet analysis outlined in [Torrence
and Compo][TC_Home] (BAMS, 1998)

[TC_home]: http://paos.colorado.edu/research/wavelets/
[TC_98]: http://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf

![random data](https://raw.github.com/aaren/wavelets/master/tests/coi_example.png)


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

### Installation ###

```sh
pip install git+https://github.com/aaren/wavelets
```

or install from a local copy:
```sh
git clone https://github.com/aaren/wavelets.git
cd wavelets
pip install -r requirements.txt
python setup.py install
# Optional: Run testsuite
pip install -r test-requirements.txt
nosetests
```

### Requirements ###

- Python 2.7+ / 3.4+ (other versions might work but have not been tested)
- Numpy (developed with 1.7.1)
- Scipy (developed with 0.12.0)

Scipy is only used for `signal.fftconvolve` and `optimize.fsolve`,
and could potentially be removed.


### License ###

The MIT License (MIT)

Copyright (c) 2013 Aaron O'Leary (dev@aaren.me)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
