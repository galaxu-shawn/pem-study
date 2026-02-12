import numpy as np
def mimo_capacity(
        channel_matrix,
        noise_cov_matrix,
        P_total,
        bandwidth,
        do_power_allocation=True,
    ):
        """Compute the mutual information I(x,y) for the model y = Hx + n.
            Where: y is the received signal vector, H is the channel matrix,
            x is the transmitted gaussian signal vector with total power P_total
            and n is the gaussian noise vector with covariance matrix noise_cov_matrix."""
        channel_matrix = np.matrix(channel_matrix)
        channel_matrix_H = np.conj(channel_matrix)
        HRnH = np.matmul(channel_matrix_H.T, np.linalg.inv(noise_cov_matrix))
        HRnH = np.matmul(HRnH, channel_matrix)

        rank = np.linalg.matrix_rank(HRnH)
        eigen_values, _ = np.linalg.eig(HRnH)
        eigen_values = np.real(eigen_values)
        non_zero_eigen_values = np.sort(eigen_values)[-rank:][::-1]

        if do_power_allocation:
            P_opt = waterfilling(non_zero_eigen_values, P_total)

        else:
            P_opt = P_total / rank

        powers = np.zeros_like(non_zero_eigen_values)
        powers[:rank] = P_opt


        mutual_information = bandwidth * np.sum(np.log2(1 + non_zero_eigen_values * powers))

        return mutual_information


def waterfilling(eigenModes, Ptotal):
    """Perform waterfilling power allocation. 
    The function returns the optimal power allocation for the given eigenmodes and total power constraint
    Waterfilling maximizes the mutual information I(x,y) for the model y = Hx + n
    with a given power constraint ||x||^2 = P_total."""
    Nchannels = eigenModes.size
    removeChannels = 0

    sortedIndexes = np.argsort(eigenModes)[::-1]
    eigenModesSorted = eigenModes[sortedIndexes]

    mu_min = 1 / eigenModesSorted[-removeChannels - 1]
    Ps = mu_min - 1 / eigenModesSorted[: Nchannels - removeChannels]
    a = 1

    while np.sum(Ps) > Ptotal and removeChannels < Nchannels:
        removeChannels += 1
        mu_min = 1 / eigenModes[-removeChannels - 1]
        Ps = mu_min - 1 / eigenModesSorted[: Nchannels - removeChannels]

    P_diff = Ptotal - np.sum(Ps)
    P_opt_aux = P_diff / (Nchannels - removeChannels) + Ps
    P_opt = np.zeros(eigenModes.size)
    P_opt[sortedIndexes[: Nchannels - removeChannels]] = P_opt_aux

    return P_opt

def siso_capacity(snr):
    """Compute the mutual information I(x,y) for the model y = x + n.
    Where: y is the received signal, x is the transmitted gaussian random variable,
    with variance Px and n is the gaussian noise with variance Pn,
    the snr is defined as snr = Px/Pn."""
    return np.log2(1 + snr)

def mse_lmmse(snr,Np):
    """Compute the mean square error of the linear minimum mean square error (LMMSE) estimator
    for the model y = h*p + n, where y is the received signal vector, h is the scalar channel to be estimated.
    p is the transmitted pilot vector of length Np signal with variance Pp, and n is the noise with variance Pn,
    the snr is given by snr = Pp/Pn.
    """
    return 1/(1+Np*snr)

def mse_ls(snr,Np):
    """Compute the mean square error of least squres (LS) estimator
    for the model y = h*p + n, where y is the received signal vector, h is the scalar channel to be estimated.
    p is the transmitted pilot vector of length Np signal with variance Pp, and n is the noise with variance Pn,
    the snr is given by snr = Pp/Pn.
    """
    mse_ls = 1/(Np*snr)
    return mse_ls
    
def snr_estimation_error(signal_power,noise_power,mse):
    """Compute the signal-to-noise ratio (SNR) for the model y = h_hat + n + e,
    where h_hat is the estimated channel, n is the noise and e is the estimation error."""
    # if mse is an array of value, replace any value greater than 1 with 1
    if isinstance(mse, np.ndarray):
        mse[mse>1] = 1
    # if mse is a scalar, replace any value greater than 1 with 1
    else:  
        if mse>1:
            return 0

    snr = signal_power*(1-mse)/(mse*signal_power + noise_power)
    return snr
