This module aims to follow the fundamental wavelet analysis outlined
in Torrence and Compo (BAMS, 1998).

### Usage ###

    :::python
    from wavelets import WaveletAnalysis
    
    # given a signal x(t)

    WT = WaveletAnalysis(x)
    wt = WT.wt()
