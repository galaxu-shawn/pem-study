import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

class OFDMChannel:
    """
    Orthogonal Frequency Division Multiplexing (OFDM) Channel Simulator.
    
    This class provides a complete OFDM communication system simulation including
    transmitter processing, channel modeling, receiver processing, and visualization.
    It implements 16-QAM modulation with pilot-assisted channel estimation and 
    supports various SNR conditions for bit error rate analysis.
    
    The OFDM system includes:
    - Serial-to-parallel conversion and bit mapping
    - 16-QAM constellation mapping with Gray coding
    - Pilot carrier insertion for channel estimation
    - IFFT/FFT operations for time/frequency domain conversion
    - Cyclic prefix addition/removal
    - Multipath channel simulation with AWGN
    - Zero-forcing channel estimation with interpolation
    - Hard decision demapping and BER calculation
    
    Attributes:
        SNRdb (float): Signal-to-Noise Ratio in dB
        K (int): Number of OFDM subcarriers
        P (int): Number of pilot carriers per OFDM block
        mu (int): Bits per symbol (modulation order, e.g., 4 for 16-QAM)
        pilotCarriers (numpy.ndarray): Indices of pilot subcarriers
        dataCarriers (numpy.ndarray): Indices of data subcarriers
        QAM_est (numpy.ndarray): Estimated QAM constellation symbols at receiver
        hardDecision (numpy.ndarray): Hard decision constellation points
        OFDM_TX (numpy.ndarray): Transmitted OFDM signal with cyclic prefix
        OFDM_RX (numpy.ndarray): Received OFDM signal after channel and noise
        mapping_table (dict): 16-QAM constellation mapping from bits to symbols
    
    Example:
        >>> # Create OFDM channel with 18dB SNR, 64 subcarriers, 8 pilots
        >>> ofdm = OFDMChannel(SNRdb=18, K=64, P=8, mu=4)
        >>> 
        >>> # Simulate channel response (example)
        >>> ch_response = np.random.randn(64) + 1j * np.random.randn(64)
        >>> 
        >>> # Run OFDM simulation
        >>> qam_est, hard_decision, ber = ofdm.ofdm_example(ch_response)
        >>> 
        >>> # Generate visualization plots
        >>> figures = ofdm.generate_plots(show_all_plots=True)
        >>> print(f"Bit Error Rate: {ber:.6f}")
    """

    def __init__(self, SNRdb=18, K=64, P=8, mu=4):
        """
        Initialize the OFDM Channel simulator.
        
        Parameters:
            SNRdb (float, optional): Signal-to-Noise Ratio in dB at the receiver.
                Higher values result in better transmission quality and lower BER.
                Typical range: 0-30 dB. Default is 18 dB.
                
            K (int, optional): Number of OFDM subcarriers (must be power of 2 for efficient FFT).
                Determines the frequency resolution and symbol duration.
                Default is 64 subcarriers.
                
            P (int, optional): Number of pilot carriers per OFDM block used for channel estimation.
                Pilots are evenly distributed across the frequency spectrum.
                Higher values improve channel estimation but reduce data throughput.
                Default is 8 pilots.
                
            mu (int, optional): Bits per symbol (modulation order). Currently supports:
                - mu=4: 16-QAM modulation (4 bits per symbol)
                This determines the constellation size (2^mu points).
                Default is 4 (16-QAM).
        
        Note:
            The cyclic prefix length is automatically set to K/4 (25% of symbol duration)
            to provide adequate protection against multipath interference.
        """
        self.SNRdb = SNRdb  # Signal-to-Noise Ratio in dB
        self.K = K  # number of OFDM subcarriers

        # The number of pilots P in the OFDM symbol describes, how many carriers are
        #   used to transmit known information (i.e. pilots). Pilots will be used at
        #   the receiver to estimate the wireless channel between transmitter and
        #   receiver. Further, we also define the value that each pilots transmits
        #   (which is known to the receiver).
        self.P = P # number of pilot carriers per OFDM block

        self.mu = mu  # bits per symbol (i.e. 16QAM)

    def ofdm_example(self,ch_response_sim):
        """
        Simulate a complete OFDM transmission through a wireless channel.
        
        This method performs the entire OFDM communication chain from bit generation
        to bit error rate calculation, including transmitter processing, channel
        simulation, and receiver processing with channel estimation.
        
        The simulation process includes:
        1. Random bit generation
        2. Serial-to-parallel conversion and QAM mapping
        3. Pilot insertion and OFDM symbol creation
        4. IFFT transformation to time domain
        5. Cyclic prefix addition
        6. Channel convolution and noise addition
        7. Cyclic prefix removal and FFT
        8. Channel estimation using pilots
        9. Zero-forcing equalization
        10. QAM demapping and bit recovery
        11. BER calculation
        
        Parameters:
            ch_response_sim (numpy.ndarray): Complex-valued channel frequency response.
                Should be a 1D array of length K representing the channel transfer
                function at each subcarrier frequency. The channel response is
                normalized internally to maintain signal power.
                
        Returns:
            tuple: A tuple containing:
                - QAM_est (numpy.ndarray): Estimated QAM constellation symbols after
                  equalization. Complex-valued array of received constellation points.
                - hardDecision (numpy.ndarray): Hard decision constellation points.
                  The closest valid constellation points to the received symbols.
                - ber (float): Bit Error Rate. Fraction of incorrectly decoded bits
                  (0.0 = perfect transmission, 1.0 = all bits wrong).
                  
        Example:
            >>> # Create a frequency-selective channel
            >>> channel = np.fft.fft([1, 0, 0.3+0.3j], 64)  # 2-tap multipath
            >>> qam_est, hard_dec, ber = ofdm.ofdm_example(channel)
            >>> print(f"BER: {ber:.6f}")
            
        Note:
            This method also populates several instance attributes (pilotCarriers,
            dataCarriers, QAM_est, etc.) that are used by the generate_plots() method.
        """
        # The length of the cyclic prefix (CP) denotes the number of samples that are
        #   copied from the end of the modulated block to the beginning, to yield a
        #   cyclic extension of the block.

        CP = self.K // 4  # length of the cyclic prefix: 25% of the block

    
        pilotValue = (self.mu - 1) * (1 + 1j)  # The known value each pilot transmits

        # Let's define the carrier sets. Some will carry data payloads, and some will
        #   carry pilot signals that will be used for channel estimation.

        allCarriers = np.arange(self.K)  # indices of all subcarriers ([0, 1, ... K-1])

        pilotCarriers = allCarriers[::self.K // self.P]  # Pilots is every (K/P)th carrier.

        # For convenience of channel estimation, let's make the last carrier also be a pilot
        pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
        P = self.P + 1

        # data carriers are all remaining carriers
        dataCarriers = np.delete(allCarriers, pilotCarriers)

        print("allCarriers:   %s" % allCarriers)
        print("pilotCarriers: %s" % pilotCarriers)
        print("dataCarriers:  %s" % dataCarriers)



        # Let's define the modulation index μ and the corresponding mapping table. We
        #   consider 16QAM transmission, i.e. we have μ=4 bits per symbol. Furthermore,
        #   the mapping from groups of 4 bits to a 16QAM constellation symbol shall be
        #   defined in mapping_table.

        payloadBits_per_OFDM = len(dataCarriers) * self.mu  # number of payload bits per OFDM symbol

        mapping_table = {
            (0, 0, 0, 0): -3 - 3j,
            (0, 0, 0, 1): -3 - 1j,
            (0, 0, 1, 0): -3 + 3j,
            (0, 0, 1, 1): -3 + 1j,
            (0, 1, 0, 0): -1 - 3j,
            (0, 1, 0, 1): -1 - 1j,
            (0, 1, 1, 0): -1 + 3j,
            (0, 1, 1, 1): -1 + 1j,
            (1, 0, 0, 0): 3 - 3j,
            (1, 0, 0, 1): 3 - 1j,
            (1, 0, 1, 0): 3 + 3j,
            (1, 0, 1, 1): 3 + 1j,
            (1, 1, 0, 0): 1 - 3j,
            (1, 1, 0, 1): 1 - 1j,
            (1, 1, 1, 0): 1 + 3j,
            (1, 1, 1, 1): 1 + 1j
        }




        # Above, we have plotted the 16QAM constellation, along with the bit-labels.
        #   Note the Gray-mapping, i.e. two adjacent constellation symbols differ only
        #   by one bit and the other 3 bits remain the same. This technique helps to
        #   minimize bit-errors, in case a wrong constellation symbol is detected: Most
        #   probably, symbol errors are "off-by-one" errors, i.e. a symbol next to the
        #   correct symbol is detected. Then, only a single bit-error occurs.

        # The demapping table is simply the inverse mapping of the mapping table:

        demapping_table = {v: k for k, v in mapping_table.items()}

        # Let us now define the wireless channel between transmitter and receiver.
        #   Here, we use a two-tap multipath channel with given impulse response
        #   channelResponse. Also, we plot the corresponding frequency response.
        #   As we see, the channel is frequency-selective. Further, we define the
        #   signal-to-noise ratio in dB, that should occur at the receiver.

        # channelResponse = np.array([1, 0, .3+0.3j])  # the impulse response of the wireless channel
        ch_min = np.min(np.abs(ch_response_sim))
        ch_max = np.max(np.abs(ch_response_sim))
        ch_p2p = np.ptp(ch_response_sim)
        amp = 1 / (ch_min + (ch_max - ch_min) / 2)
        H_exact = ch_response_sim * amp  # np.fft.fft(channelResponse, K)
        # H_exact_orin = ch_response_sim
        channelResponse = np.fft.ifft(H_exact)
        # H_exact = np.fft.ifft(H_exact,K)
        # H_exact = np.fft.fft(channelResponse, K)

        # if create_plots:
        #     plt.figure()
        #     plt.plot(allCarriers, abs(H_exact))
        #     plt.title('Wireless channel response (2-tap multi-path model before noise)')
        #     plt.xlabel('Subcarrier index (freq)')
        #     plt.ylabel('$|H(f)|$')

        # SNRdb = 18  # signal to noise-ratio in dB at the receiver (25 dB will deliver perfect transmission)

        # Now, develop and apply a random bitstream
        #  It all starts with a random bit sequence b. We generate the according bits
        #   by a random generator that draws from a Bernoulli distribution with p=0.5,
        #   i.e. 1 and 0 have equal probability. Note, that the Bernoulli distribution
        #   is a special case of the Binomial distribution, when only one draw is
        #   considered (n=1):

        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        print("Bits count: ", len(bits))
        print("First 20 bits: ", bits[:20])
        print("Mean of bits (should be around 0.5): ", np.mean(bits))

        # The bits are now sent to a serial-to-parallel converter, which groups the bits
        #   for the OFDM frame into a groups of mu bits (i.e. one group for each subcarrier):

        def SP(bits):
            return bits.reshape((len(dataCarriers), self.mu))

        bits_SP = SP(bits)
        print("First 10 bit groups")
        print(bits_SP[:10, :])

        # Now, the bits groups are sent to the mapper. The mapper converts the groups
        #   into complex-valued constellation symbols according to the mapping_table.

        def Mapping(bits):
            return np.array([mapping_table[tuple(b)] for b in bits])

        QAM = Mapping(bits_SP)
        print("First 10 QAM symbols and bits:")
        print(bits_SP[:10, :])
        print(QAM[:10])

        # The next step (which is not shown in the diagram) is the allocation of
        #   different subcarriers with data and pilots. For each subcarrier we have
        #   defined wether it carries data or a pilot by the arrays dataCarriers and
        #   pilotCarriers. Now, to create the overall OFDM data, we need to put the
        #   data and pilots into the OFDM carriers:

        def OFDM_symbol(QAM_payload):
            symbol = np.zeros(self.K, dtype=complex)  # the overall K subcarriers
            symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
            symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
            return symbol

        OFDM_data = OFDM_symbol(QAM)
        print("Number of OFDM carriers in frequency domain: ", len(OFDM_data))

        # Now, the OFDM carriers contained in OFDM_data can be transformed to the
        #   time-domain by means of the IDFT operation:
        def IDFT(OFDM_data):
            return np.fft.ifft(OFDM_data)

        OFDM_time = IDFT(OFDM_data)
        print("Number of OFDM samples in time-domain before CP: ", len(OFDM_time))

        # Subsequently, we add a cyclic prefix to the symbol. This operation
        #   concatenates a copy of the last CP samples of the OFDM time domain signal
        #   to the beginning. This way, a cyclic extension is achieved. The CP fulfills
        #   two tasks:
        #
        #   1. It isolates different OFDM blocks from each other when the wireless channel
        #      contains multiple paths, i.e. is frequency-selective.
        #   2. It turns the linear convolution with the channel into a circular one.
        #      Only with a circular convolution, we can use the single-tap equalization
        #      OFDM is so famous for.
        def addCP(OFDM_time):
            cp = OFDM_time[-CP:]  # take the last CP samples ...
            return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

        OFDM_withCP = addCP(OFDM_time)
        print("Number of OFDM samples in time domain with CP: ", len(OFDM_withCP))

        # Now, the signal is sent to the antenna and sent over the air to the receiver.
        #   In between both antennas, there is the wireless channel. We model this
        #   channel as a static multipath channel with impulse response channelResponse.
        #   Hence, the signal at the receive antenna is the convolution of the transmit
        #   signal with the channel response. Additionally, we add some noise to the
        #   signal according to the given SNR value:
        def channel(signal):
            convolved = np.convolve(signal, channelResponse)
            signal_power = np.mean(abs(convolved ** 2))
            sigma2 = signal_power * 10 ** (-self.SNRdb / 10)  # calculate noise power based on signal power and SNR

            print("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))

            # Generate complex noise with given variance
            noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
            return convolved + noise

        OFDM_TX = OFDM_withCP
        OFDM_RX = channel(OFDM_TX)


        # Rx side now, first we'll remove the CP from the signal
        def removeCP(signal):
            return signal[CP:(CP + self.K)]

        OFDM_RX_noCP = removeCP(OFDM_RX)

        # Recover our signal in the freq domain to get the received value on each subcarrier
        def DFT(OFDM_RX):
            return np.fft.fft(OFDM_RX)

        OFDM_demod = DFT(OFDM_RX_noCP)

        # The wireless channel needs to be estimated so we can apply equalization. In this
        #   case, we'll use a simple zero-forcing channel estimation followed by a simple
        #   interpolation.
        # The transmit signal contains pilot vals at certain pilot carriers. These values
        #   and their positions in the freq domain (pilot carrier index) are known to the
        #   Rx. Using the received info at the pilot carriers, the Rx can estimate the
        #   effect of the wireless channel onto this subcarrier. Therefore, the Rx gains
        #   inforamtion about the wireless channel at the pilot carriers. However, it
        #   wants to know what happened at the data carriers. In order to achieve this, it
        #   interpolates the channel values between the pilot carriers to get an estimate
        #   of the channel in the data carriers.
        def channelEstimate(OFDM_demod):
            pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
            Hest_at_pilots = pilots / pilotValue  # divide by the transmitted pilot values

            # Perform interpolation between the pilot carriers to get an estimate
            # of the channel in the data carriers. Here, we interpolate absolute value and phase
            # separately
            Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
            Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
            Hest = Hest_abs * np.exp(1j * Hest_phase)

            # if create_plots:
            #     plt.figure()
            #     plt.title('Physical Channel Estimator')
            #     plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
            #     plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
            #     plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
            #     plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
            #     plt.ylim(0,2)

            return Hest

        Hest = channelEstimate(OFDM_demod)

        # Now that we have a channel estimate at all carriers, we can use it in the channel
        #  equalizer step. Here, for each subcarrier, the influence of the channel is removed
        #  such that we get the clear (only noisy) constellation symbols back.
        def equalize(OFDM_demod, Hest):
            return OFDM_demod / Hest

        equalized_Hest = equalize(OFDM_demod, Hest)

        # Next, we extract the data carriers from teh equalized symbol. here, we throw
        #  away the pilot carriers, as they don't carry the data payload, but they were
        #  used in teh channel equalization process.
        def get_payload(equalized):
            return equalized[dataCarriers]

        QAM_est = get_payload(equalized_Hest)



        # Now, that the constellation is obtained back, we need to send the complex
        #   values to the demapper, to transform the constellation points to the bit
        #   groups. In order to do this, we compare each received constellation point
        #   against each possible constellation point and choose the constellation point
        #   which is closest to the received point. Then, we return the bit-group that
        #   belongs to this point.
        def Demapping(QAM):
            # array of possible constellation points
            constellation = np.array([x for x in demapping_table.keys()])

            # calculate distance of each RX point to each possible point
            dists = abs(QAM.reshape((-1, 1)) - constellation.reshape((1, -1)))

            # for each element in QAM, choose the index in constellation
            # that belongs to the nearest constellation point
            const_index = dists.argmin(axis=1)

            # get back the real constellation point
            hardDecision = constellation[const_index]

            # transform the constellation point into the bit groups
            return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

        PS_est, hardDecision = Demapping(QAM_est)



        # Finally, the bit groups need to be converted to a serial stream of bits, by means
        #  of a parallel to serial conversion
        def PS(bits):
            return bits.reshape((-1,))

        bits_est = PS(PS_est)

        #  Now that all bits are decoded, we can calculate the BER:
        print("Obtained Bit error rate: ", np.sum(abs(bits - bits_est)) / len(bits))
        ber = np.sum(abs(bits - bits_est)) / len(bits)

        self.pilotCarriers = pilotCarriers
        self.dataCarriers = dataCarriers
        self.QAM_est = QAM_est
        self.hardDecision = hardDecision
        self.OFDM_TX = OFDM_TX
        self.OFDM_RX = OFDM_RX
        self.mapping_table = mapping_table

        return QAM_est, hardDecision, ber


    def generate_plots(self, show_all_plots=False):
        """
        Generate comprehensive visualization plots of the OFDM system performance.
        
        Creates multiple matplotlib figures showing different aspects of the OFDM
        transmission and reception process. All figures are returned in a dictionary
        for programmatic access, and can optionally be displayed immediately.
        
        The generated plots include:
        1. Pilot and data carrier allocation in frequency domain
        2. 16-QAM constellation diagram with Gray coding
        3. Transmitted vs received signal magnitude comparison
        4. Received constellation after channel and noise
        5. Hard decision demapping visualization
        6. Final received constellation points
        
        Parameters:
            show_all_plots (bool, optional): If True, displays all plots immediately
                using plt.show(). If False, plots are created but not displayed,
                allowing for programmatic manipulation. Default is False.
                
        Returns:
            dict: Dictionary containing matplotlib figure objects with the following keys:
                - 'pilot_data_carriers': Subcarrier allocation plot
                - 'qam_constellation': Ideal 16-QAM constellation with bit labels
                - 'tx_rx_signals': Time domain signal magnitude comparison
                - 'received_constellation': Received symbols before hard decision
                - 'hard_decision_demapping': Visualization of symbol decision process
                - 'final_received_constellation': Final decoded constellation points
                
        Example:
            >>> # Generate plots without displaying
            >>> figures = ofdm.generate_plots()
            >>> 
            >>> # Access individual figures
            >>> constellation_fig = figures['qam_constellation']
            >>> constellation_fig.savefig('constellation.png')
            >>> 
            >>> # Generate and display all plots
            >>> figures = ofdm.generate_plots(show_all_plots=True)
            
        Note:
            This method requires that ofdm_example() has been called first to populate
            the necessary data attributes (pilotCarriers, dataCarriers, QAM_est, etc.).
            
        Raises:
            AttributeError: If called before ofdm_example(), required attributes
                may not exist.
        """
        plt.clf()
        figures = {}
        mu = self.mu
        
        # OFDM Pilot and Data Carriers plot
        fig1 = plt.figure(figsize=(8,2))
        plt.plot(self.pilotCarriers, np.zeros_like(self.pilotCarriers), 'bo', label='pilot')
        plt.plot(self.dataCarriers, np.zeros_like(self.dataCarriers), 'ro', label='data')
        plt.title('OFDM Pilot (blue) and Data (red) Carriers')
        plt.xlabel('Subcarrier (freq)')
        plt.grid(linestyle=':')
        figures['pilot_data_carriers'] = fig1

        # QAM constellation plot
        fig2 = plt.figure()
        for b3 in [0, 1]:
            for b2 in [0, 1]:
                for b1 in [0, 1]:
                    for b0 in [0, 1]:
                        B = (b3, b2, b1, b0)
                        Q = self.mapping_table[B]
                        plt.plot(Q.real, Q.imag, 'bo')
                        plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')
        plt.title('QAM signaling constellation with Gray-mapping')
        plt.xlim([-mu,mu])
        plt.ylim([-mu,mu])
        plt.grid(linestyle=':')
        figures['qam_constellation'] = fig2

        # TX/RX signal comparison
        fig3 = plt.figure(figsize=(8,2))
        plt.plot(abs(self.OFDM_TX), label='TX signal')
        plt.plot(abs(self.OFDM_RX), label='RX signal')
        plt.legend(fontsize=10)
        plt.xlabel('Time'); plt.ylabel('$|x(t)|$')
        plt.grid(True)
        figures['tx_rx_signals'] = fig3

        # Received constellation
        fig4 = plt.figure()
        plt.plot(self.QAM_est.real, self.QAM_est.imag, 'bo')
        plt.title('Received constellation')
        plt.xlim([-mu,mu])
        plt.ylim([-mu,mu])
        plt.grid(linestyle=':')
        figures['received_constellation'] = fig4

        # Hard Decision demapping
        fig5 = plt.figure()
        for qam, hard in zip(self.QAM_est, self.hardDecision):
            plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o');
            plt.plot(self.hardDecision.real, self.hardDecision.imag, 'ro')
        plt.title('Hard Decision demapping')
        plt.xlim([-mu,mu])
        plt.ylim([-mu,mu])
        plt.grid(linestyle=':')
        figures['hard_decision_demapping'] = fig5

        # Final received constellation (removing duplicate)
        fig6 = plt.figure()
        plt.plot(self.QAM_est.real, self.QAM_est.imag, 'bo');
        plt.title('Final Received constellation')
        plt.xlim([-mu,mu])
        plt.ylim([-mu,mu])
        plt.grid(linestyle=':')
        figures['final_received_constellation'] = fig6

        if show_all_plots:
            plt.show()

        return figures
