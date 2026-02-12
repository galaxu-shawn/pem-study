import os
import sys
import numpy as np
import scipy.interpolate
from scipy.stats import linregress
from numpy import pi, arccosh, sqrt, cos
from scipy.fftpack import fftshift, fft2, ifft2, fft, ifft
from scipy.signal import firwin, filtfilt


from pem_utilities.utils import apply_math_function



#all FT's assumed to be centered at the origin
def ft(f, ax=-1):
    F = fftshift(fft(fftshift(f), axis = ax))
    
    return F
    
def ift(F, ax = -1):
    f = fftshift(ifft(fftshift(F), axis = ax))
    
    return f

def ft2(f, delta=1):
    F = fftshift(fft2(fftshift(f)))*delta**2
    
    return(F)

def ift2(F, delta=1):
    N = F.shape[0]
    f = fftshift(ifft2(fftshift(F)))*(delta*N)**2
    
    return(f)

def RECT(t,T):
    f = np.zeros(len(t))
    f[(t/T<0.5) & (t/T >-0.5)] = 1
    
    return f
    
def taylor_win(nsamples, S_L=43):
    xi = np.linspace(-0.5, 0.5, nsamples)
    A = 1.0/pi*arccosh(10**(S_L*1.0/20))
    n_bar = int(2*A**2+0.5)+1
    sigma_p = n_bar/sqrt(A**2+(n_bar-0.5)**2)
    
    #Compute F_m
    m = np.arange(1,n_bar)
    n = np.arange(1,n_bar)
    F_m = np.zeros(n_bar-1)
    for i in m:
        num = 1
        den = 1
        for j in n:
            num = num*\
            (-1)**(i+1)*(1-i**2*1.0/sigma_p**2/(\
                            A**2+(j-0.5)**2))
            if i!=j:
                den = den*(1-i**2*1.0/j**2)
            
        F_m[i-1] = num/den
    
    w = np.ones(nsamples)
    for i in m:
        w += F_m[i-1]*cos(2*pi*i*xi)
    
    w = w/w.max()          
    return(w)
    
def upsample(f,size):
    x_pad = size[1]-f.shape[1]
    y_pad = size[0]-f.shape[0]
    
    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0
        
    if np.mod(y_pad,2):
        y_off = 1
    else:
        y_off = 0
    
    F = ft2(f)
    F_pad = np.pad(F, ((y_pad/2,y_pad/2+y_off),(x_pad/2, x_pad/2+x_off)),
                   mode = 'constant')
    f_up = ift2(F_pad)
    
    return(f_up)
    
def upsample1D(f, size):
    x_pad = size-f.size
    
    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0
    
    F = ft(f)
    F_pad = np.pad(F, (x_pad/2, x_pad/2+x_off),
                   mode = 'constant')
    f_up = ift(F_pad)
    
    return(f_up)
    
def pad1D(f, size):
    x_pad = size-f.size
    
    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0
    
    
    f_pad = np.pad(f, (x_pad/2, x_pad/2+x_off),
                   mode = 'constant')
    
    return(f_pad)

def pad(f, size):
    x_pad = size[1]-f.shape[1]
    y_pad = size[0]-f.shape[0]
    
    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0
        
    if np.mod(y_pad,2):
        y_off = 1
    else:
        y_off = 0
    
    f_pad = np.pad(f, ((y_pad//2,y_pad//2+y_off),(x_pad//2, x_pad//2+x_off)),
                   mode = 'constant')
    
    return(f_pad)
    
def cart2sph(cart):
    x = np.array([cart[:,0]]).T
    y = np.array([cart[:,1]]).T
    z = np.array([cart[:,2]]).T
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    sph = np.hstack([azimuth, elevation, r])
    return sph
    
def sph2cart(sph):
    azimuth     = np.array([sph[:,0]]).T
    elevation   = np.array([sph[:,1]]).T
    r           = np.array([sph[:,2]]).T
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    cart = np.hstack([x,y,z])
    return cart
    
def decimate(x, q, n=None, axis=-1, beta = None, cutoff = 'nyq'):
    if not isinstance(q, int):
        raise TypeError("q must be an integer")
        
    if n == None:
        n = int(np.log2(x.shape[axis]))
        
    if x.shape[axis] < n:
        n = x.shape[axis]-1
    
    if beta == None:
        beta = 1.*n/8
    
    padlen = n/2
    
    if cutoff == 'nyq':
        eps = np.finfo(np.float).eps
        cutoff = 1.-eps
    
    window = ('kaiser', beta)
    a = 1.
    
    b = firwin(n,  cutoff/ q, window=window)
    y = filtfilt(b, [int(a)], x, axis=axis, padlen = int(padlen))
    
    sl = [slice(None)] * y.ndim
    sl[axis] = slice(None, None, q)
    return y[tuple(sl)]

def window_function(function='Flat', size=512, dbDown=30):
    if function.lower() == 'hann':
        win = np.hanning(size)
    elif function.lower() == 'hamming':
        win = np.hamming(size)
    else:
        win = np.ones(size)
    win_sum = np.sum(win)
    win *= size/win_sum
    return win, win_sum

def platform_dict(xyz_domain, freq_domain=None, bandwidth=None, num_freqs=None, num_pulse_CPI=None):
    c = 299792458.0
    nsamples = num_freqs
    npulses = num_pulse_CPI
    freq = freq_domain
    pos = xyz_domain
    k_r  = 4*np.pi*freq/c
    B_IF = bandwidth
    delta_r = c/(2*B_IF)
    delta_t = 1.0/B_IF
    t = np.linspace(-nsamples/2, nsamples/2, nsamples)*delta_t
        
    chirprate, f_0, r, p, s = linregress(t, freq)
                    
    #Vector to scene center at synthetic aperture center
    if np.mod(npulses,2)>0:
        R_c = pos[int(npulses/2)]
    else:
        R_c = np.mean(
                pos[int(npulses/2)-1:int(npulses/2)+1],
                axis = 0)
        
    #Save values to dictionary for export
    platform = \
    {
        'f_0'       :   f_0,
        'freq'      :   freq,
        'chirprate' :   chirprate,
        'B_IF'      :   B_IF,
        'nsamples'  :   nsamples,
        'npulses'   :   npulses,
        'pos'       :   pos,
        'delta_r'   :   delta_r,
        'R_c'       :   R_c,
        't'         :   t,
        'k_r'       :   k_r,
    }

                
                    
    if np.mod(npulses,2)>0:
        R_c = pos[int(npulses/2)]
    else:
        R_c = np.mean(
                pos[int(npulses/2)-1:int(npulses/2)+1],
                axis = 0)
                    
    
    # Synthetic aperture length
    L = np.linalg.norm(pos[-1]-pos[0])
    # #Add k_y
    platform['k_y'] = np.linspace(-npulses/2,npulses/2,npulses)*2*np.pi/L   

    # convert cartesian coordinates to spherical coordinates
    platform['pos_spherical'] = np.zeros_like(pos)
    platform['pos_spherical'][:,0] = np.linalg.norm(pos, axis=1)  # radius
    platform['pos_spherical'][:,1] = np.arctan2(pos[:,1], pos[:,0])  # azimuth
    platform['pos_spherical'][:,2] = np.arcsin(pos[:,2] / platform['pos_spherical'][:,0])  # elevation


    return platform

def img_plane_dict(platform, res_factor=1.0, n_hat = np.array([0,0,1]), aspect = 1, upsample = True):
    """
    
     Image Plane Dictionary 
     This function defines the image plane parameters.  The user specifies the 
     image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  
     image plane whose pixels are sized at the theoretical resolution limit    
     of the system (derived using delta_r which in turn was derived using the  
     bandwidth.  The user can also adjust the aspect of the image grid.  This  
     defaults to nsamples/npulses.                                             
                                                                               
     'n_hat' is a user specified value that defines the image plane            
     orientation w.r.t. to the nominal ground plane.                           
                                                                               
    """
    nsamples = platform['nsamples']
    npulses = platform['npulses']
    
    #Import relevant platform parameters
    R_c = platform['R_c']    
    
    #Define image plane parameters
    if upsample:
        nu= 2**int(np.log2(nsamples)+bool(np.mod(np.log2(nsamples),1)))
        nv= 2**int(np.log2(npulses)+bool(np.mod(np.log2(npulses),1)))
    else:
        nu= nsamples
        nv= npulses
        
    #Define resolution.  This should be less than the system resolution limits
    du = platform['delta_r']*res_factor*nsamples/nu
    dv = aspect*du
    
    #Define range and cross-range locations
    u = np.arange(-nu/2, nu/2)*du
    v = np.arange(-nv/2, nv/2)*dv
    
    #Derive image plane spatial frequencies
    k_u = 2*np.pi*np.linspace(-1.0/(2*du), 1.0/(2*du), nu)
    k_v = 2*np.pi*np.linspace(-1.0/(2*dv), 1.0/(2*dv), nv)
    
    #Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c)/np.linalg.norm(np.cross(n_hat, R_c))
    u_hat = np.cross(v_hat, n_hat)/np.linalg.norm(np.cross(v_hat, n_hat))
    
    #Represent u and v in (x,y,z)
    [uu,vv] = np.meshgrid(u,v)
    uu = uu.flatten(); vv = vv.flatten()
    
    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T 
            )))            
    b = np.asmatrix(np.vstack((uu,vv)))
    pixel_locs = np.asarray(A*b)
    
    #Construct dictionary and return to caller
    img_plane =\
    {
    'n_hat'     :   n_hat,
    'u_hat'     :   u_hat,
    'v_hat'     :   v_hat,
    'du'        :   du,
    'dv'        :   dv,
    'u'         :   u,
    'v'         :   v,
    'k_u'       :   k_u,
    'k_v'       :   k_v,
    'pixel_locs':   pixel_locs # 3 x N_pixel array specifying x,y,z location
                               # of each pixel
    }
    
    return(img_plane)



def polar_format(phs, platform, img_plane, taylor = 20):
    """
    Polar Format Algorithm for Synthetic Aperture Radar (SAR) image formation.
    
    This function implements the Polar Format Algorithm (PFA) for SAR image reconstruction.
    The phase history data is collected on a two-dimensional surface in k-space, where each 
    pulse represents a strip of this surface. The algorithm projects each strip onto the 
    (ku,kv) plane defined by the normal vector in the image plane dictionary, resulting in 
    unevenly spaced data in (ku,kv). This data is then interpolated onto an evenly spaced 
    (ku,kv) grid through a two-step process: first radial interpolation, then along-track 
    interpolation.
    
    Parameters
    ----------
    phs : numpy.ndarray
        Complex phase history data array with shape (npulses, nsamples).
        Contains the received radar signal data for each pulse and frequency sample.
    
    platform : dict
        Platform parameters dictionary containing:
        - 'npulses' : int, number of radar pulses
        - 'f_0' : float, center frequency (Hz)
        - 'pos' : numpy.ndarray, platform positions for each pulse
        - 'k_r' : numpy.ndarray, radial wavenumber values
        - 'R_c' : numpy.ndarray, vector to scene center at synthetic aperture center
    
    img_plane : dict
        Image plane parameters dictionary containing:
        - 'n_hat' : numpy.ndarray, normal vector defining image plane orientation
        - 'k_u' : numpy.ndarray, spatial frequencies in u-direction
        - 'k_v' : numpy.ndarray, spatial frequencies in v-direction
    
    taylor : float, optional
        Taylor window sidelobe level in dB for sidelobe suppression (default: 20).
        Applied to both range and azimuth dimensions to reduce imaging artifacts.
    
    Returns
    -------
    numpy.ndarray
        Real-valued SAR image formed using the polar format algorithm.
        The image represents the magnitude of the reconstructed radar reflectivity.
    
    Notes
    -----
    The Polar Format Algorithm consists of the following key steps:
    1. Project k-space data strips onto the (ku,kv) plane using the image plane normal vector
    2. Compute wavenumber offset based on the grazing angle
    3. Perform radial interpolation to map polar k-space data to rectangular grid
    4. Apply Taylor windowing for sidelobe reduction
    5. Perform along-track interpolation to complete the rectangular grid
    6. Apply 2D Fourier transform to obtain the final image
    
    The algorithm assumes that the radar data has been properly motion compensated and
    that the platform trajectory is approximately linear over the synthetic aperture.
    
    References
    ----------
    Further details of this method are given in:
    - Jakowatz, C.V., et al. "Spotlight-Mode Synthetic Aperture Radar"
    - Carrara, W.G., et al. "Spotlight Synthetic Aperture Radar"
    """

    #Retrieve relevent parameters
    c           =   299792458.0
    npulses     =   platform['npulses']
    f_0         =   platform['f_0']
    pos         =   np.asarray(platform['pos'])
    k           =   platform['k_r']
    R_c         =   platform['R_c']
    n_hat       =   img_plane['n_hat']
    k_ui        =   img_plane['k_u']
    k_vi        =   img_plane['k_v']
    
    #Compute k_xi offset
    psi = np.pi/2-np.arccos(np.dot(R_c,n_hat)/np.linalg.norm(R_c))
    k_ui = k_ui + 4*np.pi*f_0/c*np.cos(psi)
    
    #Compute number of samples in scene
    nu = k_ui.size
    nv = k_vi.size
    
    #Compute x and y unit vectors. x defined to lie along R_c.
    #z = cross(vec[0], vec[-1]); z =z/np.linalg.norm(z)
    u_hat = (R_c-np.dot(R_c,n_hat)*n_hat)/\
            np.linalg.norm((R_c-np.dot(R_c,n_hat)*n_hat))
    v_hat = np.cross(u_hat,n_hat)
    
    #Compute r_hat, the diretion of k_r, for each pulse
    r_norm = np.linalg.norm(pos,axis=1)
    r_norm = np.array([r_norm]).T
    r_norm = np.tile(r_norm,(1,3))
    
    r_hat = pos/r_norm
    
    #Convert to matrices to make projections easier
    r_hat = np.asmatrix(r_hat)
    u_hat = np.asmatrix([u_hat])
    v_hat = np.asmatrix([v_hat])
    
    k_matrix = np.tile(k,(npulses,1))
    k_matrix = np.asmatrix(k)
    
    #Compute kx and ky meshgrid
    ku = r_hat*u_hat.T*k_matrix; ku = np.asarray(ku)
    kv = r_hat*v_hat.T*k_matrix; kv = np.asarray(kv)
    
    #Create taylor windows
    win1 = taylor_win(int(phs.shape[1]), S_L = taylor)
    win2 = taylor_win(int(phs.shape[0]), S_L = taylor)
    
    #Radially interpolate kx and ky data from polar raster
    #onto evenly spaced kx_i and ky_i grid for each pulse
    real_rad_interp = np.zeros([npulses,nu])
    imag_rad_interp = np.zeros([npulses,nu])
    ky_new = np.zeros([npulses,nu])
    for i in range(npulses):
        # print('range interpolating for pulse %i'%(i+1))
        real_rad_interp[i,:] = np.interp(k_ui, ku[i,:], 
            phs.real[i,:]*win1, left = 0, right = 0)
        imag_rad_interp[i,:] = np.interp(k_ui, ku[i,:], 
            phs.imag[i,:]*win1, left = 0, right = 0)
        ky_new[i,:] = np.interp(k_ui, ku[i,:], kv[i,:])  
    
    #Interpolate in along track direction to obtain polar formatted data
    real_polar = np.zeros([nv,nu])
    imag_polar = np.zeros([nv,nu])
    isSort = (ky_new[int(npulses/2), int(nu/2)] < ky_new[int(npulses/2)+1, int(nu/2)])
    if isSort:
        for i in range(nu):
            # print('cross-range interpolating for sample %i'%(i+1))
            real_polar[:,i] = np.interp(k_vi, ky_new[:,i], 
                real_rad_interp[:,i]*win2, left = 0, right = 0)
            imag_polar[:,i] = np.interp(k_vi, ky_new[:,i], 
                imag_rad_interp[:,i]*win2, left = 0, right = 0)
    else:
        for i in range(nu):
            # print('cross-range interpolating for sample %i'%(i+1))
            real_polar[:,i] = np.interp(k_vi, ky_new[::-1,i], 
                real_rad_interp[::-1,i]*win2, left = 0, right = 0)
            imag_polar[:,i] = np.interp(k_vi, ky_new[::-1,i], 
                imag_rad_interp[::-1,i]*win2, left = 0, right = 0)
    
    real_polar = np.nan_to_num(real_polar)
    imag_polar = np.nan_to_num(imag_polar)    
    phs_polar = np.nan_to_num(real_polar+1j*imag_polar)

    img = np.abs(ft2(phs_polar))
    
    return(img)


def isar_3d_interpolate_ST_wrapper(x, y, z, d, xi, yi, zi, q):
    test_w = scipy.interpolate.griddata((x, y, z), d, (xi, yi, zi), method='nearest', fill_value=0.)
    q.put(test_w)

def isar_3d(data ,freq_domain,phi_domain,theta_domain, function='abs', size=(256,256,256) ,window=None):
    freqs = freq_domain
    nfreq = len(freqs)
    freqs = freqs.reshape(-1,1,1)
    phis = phi_domain
    nphi = len(phis)
    phis = phis.reshape(1,-1,1)
    thetas = theta_domain
    ntheta = len(thetas)
    thetas = thetas.reshape(1,1,-1)
    
    ndrng = size[0]
    nxrng1 = size[1]
    nxrng2 = size[2]

    cel = 1
    sel = 0
    az = -phis
    az = np.unwrap(np.radians(az))
    az_ctr = np.mean(az)
    az_slant = az - az_ctr
    az_slant = az*cel
    az_slant = az_slant + az_ctr

    caz = 1
    saz = 0
    el = 90.-thetas
    el = np.unwrap(np.radians(el))
    el_ctr = np.mean(el)
    el_slant = el - el_ctr
    el_slant = el*caz
    el_slant = el_slant + el_ctr

    # generate inv synthetic aperture SAR image from frequency response vs. azimuth
    #
    # Notes: 
    #  1) We assume that the frequency data is polar samples on a plane. f and
    #     az measure the range and angle on that plane.  The image is formed on
    #     the same plane.  Usually this is the slant plane.  Note for slant
    #     plane images, the azimuth angle measured on the plane is related to
    #     the azimuth angle measured from the xy-plane by:
    #         az_slant = az_ground*cos(el)
    #     where el is the elevation angle of the slant plane.
    #
    #  2) To get a ground plane image, replace x by x*cos(el).
    #
    #  3) By default we interpolate to a rectangle whose fx values range from
    #     min(f) to max(f) and whose fy values range from about the fy extent in
    #     the middle of the annulus region.
    #
    #  4) We use cubic interpolation - search for cubic to change to another type
    #
    #  5) Down range resolution Rd = c/(2 f_span), Pd = Nf*Rd
    #     Cross range resolution Rc = lambda_c/(4*sin(az_span/2)), Pc = Na*Rc
    #
    #  6) Since img comes back as (Nx x Ny) and not (Ny x Nx), to form a
    #     rectified image with x-axis along the horizontal and y-axis
    #     pointing up, use something like these commands:
    #       imagesc(y,x,amp2db(transpose(img));
    #       axis xy;
    #
    #  7) Positive cross-range points in spatial direction opposite to
    #     positive progression of az_deg
    #
    #  Randy Moses, 8/22/2002; revised 10/21/2002
    #  Cosmetic changes by R Kipp

    # We assume the az vector goes from a minimum to a maximum angle
    az_span = np.max(az) - np.min(az)
    el_span = np.max(el) - np.min(el)

    # interpolate in the rectangular domain

    fxtrue = freqs*np.cos(az - az_ctr)*np.cos(el-el_ctr) #  % true fx and fy locations for this f, az grid
    fytrue = freqs*np.sin(az - az_ctr)*np.cos(el-el_ctr)
    fztrue = -freqs*np.ones(az.shape)*np.sin(el-el_ctr)

    # switch lower(gridsize)
    #  case {'in','inside'}
    #   %
    #   % interpolate a rectangle to the inside the populated arc
    #   %
    #   fxmin = max(fxtrue(1,:));
    #   fxmax = min(fxtrue(nf,:));
    #   fymax = max(fytrue(1,:));

    #  case {'out','outside'}
    #   %
    #   % interpolate a rectangle to the outside the populated arc
    #   %
    #   fxmin = min(fxtrue(1,:));
    #   fxmax = max(fxtrue(nf,:));
    #   fymax = max(fytrue(nf,:));

    #  case {'mid','middle'}
    #   %
    #   % interpolate a rectangle to the middle of the populated arc
    #   %
    #   fxmin = min(f_ghz);
    #   fxmax = max(f_ghz);
    #   f_c = mean(f_ghz);  % center frequency is the mean of the frequency vector
    #   fymax = sin(az_span/2)*f_c;

    # implement inside version
    fxmin = np.max(np.max(fxtrue[0,:,:]))
    fxmax = np.min(np.min(fxtrue[-1,:,:]))
    fymax = np.min(np.max(fytrue[0,:,:],axis=0))
    fzmax = np.min(np.max(fztrue[1,:,:],axis=1))

    fxtrue = fxtrue.reshape(-1)
    fytrue = fytrue.reshape(-1)
    fztrue = fztrue.reshape(-1)  
    

    fx = np.linspace(fxmin,fxmax,nfreq) # desired downrange frequencies
    fy = np.linspace(-fymax,fymax,nphi) # desired crossrange frequencies
    fz = np.linspace(-fzmax,fzmax,ntheta) # desired crossrange frequencies
    grid_y, grid_x, grid_z = np.meshgrid(fy,fx,fz)
    # grid_x = grid_x.transpose(1,0,2)
    # grid_y = grid_y.transpose(1,0,2)
    # grid_z = grid_z.transpose(1,0,2)
    shape = grid_x.shape

    rdata = scipy.interpolate.griddata((fxtrue,fytrue,fztrue), data.flatten(), (grid_x,grid_y,grid_z), method='nearest', fill_value=0.)

    # # dataf = data.to_numpy().flatten()
    # # 
    # # polar to rectangular interpolation
    # # 

    # nthreads = multiprocessing.cpu_count()
    # jump = int(grid_x.shape[2]/nthreads)

    # queue, process = [], []
    # for n in range(nthreads):
    #     queue.append(multiprocessing.Queue())
    #     # Handle the possibility that size is not evenly divisible by nprocs
    #     if n == (nthreads-1):
    #         finalidx = grid_x.shape[2]
    #     else:
    #         finalidx = (n + 1) * jump
    #     # Define the arguments, dividing the interpolation variables into
    #     # nprocs roughly evenly sized slices
    #     argtuple = (fxtrue,fytrue,fztrue,data,grid_x[:,:,(n*jump):finalidx],grid_y[:,:,(n*jump):finalidx],grid_z[:,:,(n*jump):finalidx], queue[-1])
    #     # Create the processes, and launch them
    #     process.append(multiprocessing.Process(target=isar_3d_interpolate_ST_wrapper, args=argtuple))
    #     try:
    #         process[-1].start()
    #     except Exception as e:
    #         pass

    # # Read the individual results back from the queues and concatenate them
    # # into the return array

    # for i, (q, p) in enumerate(zip(queue, process)):
    #     if i == 0:
    #         rdata = q.get()
    #     else:
    #         rdata = np.concatenate((rdata, q.get()), axis=2)
    #     p.join()

    # rdata = rdata.reshape(shape[::-1])
    # rdata = rdata.transpose(2,1,0)

    # 
    # add windowing
    # 
    winx, winx_sum = window_function(function=window, size=nfreq)
    winy, winy_sum = window_function(function=window, size=nphi)
    winz, winz_sum = window_function(function=window, size=ntheta)
    winx = winx.reshape(-1,1,1)
    winy = winy.reshape(1,-1,1)
    winz = winz.reshape(1,1,-1)

    # %
    # % zero padding
    # %
    if ndrng < nfreq:
      # warning('nx should be at least as large as the length of f -- increasing nx');
      ndrng = nfreq;
    if nxrng1 <nphi:
      # warning('ny should be at least as large as the length of az -- increasing ny');
      nxrng1 = nphi
    if nxrng2 <ntheta:
      # warning('nz should be at least as large as the length of el -- increasing nz');
      nxrng2 = ntheta

    iq = np.zeros((ndrng, nxrng1, nxrng2),dtype=np.complex_)
    xshift = (ndrng-nfreq)//2
    yshift = (nxrng1-nphi)//2
    zshift = (nxrng2-ntheta)//2
    window = winx*winy*winz
    iq[xshift:xshift+nfreq,yshift:yshift+nphi,zshift:zshift+ntheta] = np.multiply(rdata,window)
    #
    # normalize so that unit amplitude scatterers have about unit amplitude in
    # the image (normalized for the windows).  The "about" comes because of the
    # truncation of the polar shape into a rectangular shape.
    #
    iq = np.fft.fftshift(iq)*ndrng*nxrng1*nxrng2/winx_sum/winy_sum/winz_sum;     
    isar_image = np.fft.fftshift(np.fft.ifftn(iq)) # Nx x Ny
    isar_image = apply_math_function(isar_image, function)
    isar_image = isar_image[::-1,:,:]
    # 
    #  compute the image plane downrange and crossrange distance vectors (in
    #  meters)
    # 
    dfx = fx[1] - fx[0] # difference in x-frequencies
    dfy = fy[1] - fy[0] # difference in y-frequencies
    dfz = fz[1] - fz[0] # difference in z-frequencies
    c0 = 299792458.0 # speed of light in m/s
    dx = c0/(2*dfx)/ndrng
    dy = c0/(2*dfy)/nxrng1
    dz = c0/(2*dfz)/nxrng2
    x = np.arange(start=0, step=dx, stop=ndrng*dx)
    y = np.arange(start=0, step=dy, stop=nxrng1*dy)
    z = np.arange(start=0, step=dz, stop=nxrng2*dz)
    # 
    #  make the center x and y values zero
    # 
    #  these two lines ensure one value of x and y are zero when nx or ny are even
    # 
    range_values = x - x[ndrng//2]
    cross_range1_values = y - y[nxrng1//2]
    cross_range2_values = z - z[nxrng2//2]

    XR1, RR, XR2 = np.meshgrid(cross_range1_values, range_values, cross_range2_values)

    return XR1,RR,XR2,isar_image
    # # RR = RR.transpose(1,0,2)
    # # XR1 = XR1.transpose(1,0,2)
    # # XR2 = XR2.transpose(1,0,2)

    # indexes = zip(RR.flatten(order='F'), XR1.flatten(order='F'), XR2.flatten(order='F'))
    # try:
    #     return get_new_dataframe(values=isar_image.flatten(order='F'), indexes=indexes, index_names=data_type_index_names['3D ISAR'])
    # except Exception as e:
    #     pass

def omega_k(phs, platform, taylor = 20, upsample = 6):
    """
    Omega-K Algorithm for Synthetic Aperture Radar (SAR) image formation.
    
    This function implements the Omega-K (also known as Range-Doppler or wavenumber domain) 
    algorithm for SAR image reconstruction. The algorithm is based on the method prescribed 
    in the Carrara text and only requires phase history and platform parameters as inputs - 
    no image plane dictionary is needed. The algorithm performs range curvature correction 
    through matched filtering and Stolt interpolation to map the data from polar to 
    rectangular coordinates in the wavenumber domain.
    
    Parameters
    ----------
    phs : numpy.ndarray
        Complex phase history data array with shape (npulses, nsamples).
        The input data must be demodulated to a fixed reference (preferably scene center).
        Contains the received radar signal data for each pulse and frequency sample.
    
    platform : dict
        Platform parameters dictionary containing:
        - 'k_r' : numpy.ndarray, radial wavenumber values
        - 'k_y' : numpy.ndarray, along-track wavenumber values  
        - 'pos' : numpy.ndarray, platform positions for each pulse
        - 'nsamples' : int, number of range samples per pulse
        - 'npulses' : int, number of radar pulses
    
    taylor : float, optional
        Taylor window sidelobe level in dB for sidelobe suppression (default: 20).
        Applied to both range and azimuth dimensions to reduce imaging artifacts.
    
    upsample : int, optional
        Upsampling factor for zero-padding in the final image formation (default: 6).
        Controls the interpolation and final image size through spectral zero-padding.
    
    Returns
    -------
    numpy.ndarray
        Complex-valued SAR image formed using the omega-k algorithm.
        The image represents the reconstructed radar reflectivity in the spatial domain.
    
    Notes
    -----
    The Omega-K Algorithm consists of the following key steps:
    1. Perform 1D Fourier Transform along azimuth (slow-time) dimension
    2. Apply matched filter to compensate range curvature for scatterers at minimum range R_s
    3. Perform Stolt interpolation to map from polar (K_r, K_y) to rectangular (K_x, K_y) grid
    4. Apply Taylor windowing for sidelobe reduction
    5. Zero-pad the spectrum for upsampling
    6. Apply 2D inverse Fourier transform to obtain the final image
    
    Algorithm Assumptions:
    - Straight-line flight path over the synthetic aperture
    - Phase history demodulated to a fixed reference (preferably scene center)
    - Small spotlight or stripmap SAR geometry
    
    The default minimum range R_s is set to the minimum range of the platform trajectory 
    to scene center. Range curvature correction is optimized for scatterers at this range,
    with residual errors for scatterers at other ranges corrected through Stolt interpolation.
    
    References
    ----------
    Further details of this method are given in:
    - Carrara, W.G., et al. "Spotlight Synthetic Aperture Radar: Signal Processing Algorithms"
    - Cumming, I.G. and Wong, F.H. "Digital Processing of Synthetic Aperture Radar Data"
    """

    #Retrieve relevent parameters
    K_r     =   platform['k_r']
    K_y     =   platform['k_y']
    R_s     =   np.linalg.norm(platform['pos'], axis = -1).min()
    nsamples=   platform['nsamples']
    npulses =   platform['npulses']
    
    #Take azimuth FFT
    S_Kx_Kr = ift(phs, ax = 0)
    
    #Create K_r, K_y grid
    [K_r, K_y] = np.meshgrid(K_r, K_y)
    
    #Mathed filter for compensating range curvature
    phase_mf = -R_s*K_r + R_s*np.sqrt(K_r**2-K_y**2)
    phase_mf = np.nan_to_num(phase_mf)
    S_Kx_Kr_mf = S_Kx_Kr*np.exp(1j*phase_mf)
    
    #Stolt interpolation
    K_xi_max = np.nan_to_num(np.sqrt(K_r**2-K_y**2)).max()
    K_xi_min = np.nan_to_num(np.sqrt(K_r**2-K_y**2)).min()
    K_xi = np.linspace(K_xi_min, K_xi_max, nsamples)
    
    S = np.zeros([npulses,nsamples])+0j
    for i in range(npulses):
        K_x = np.nan_to_num(np.sqrt(K_r[i,:]**2-K_y[i,:]**2))
        
        f_real = scipy.interpolate.interp1d(K_x, S_Kx_Kr_mf[i,:].real, kind = 'linear', bounds_error = 0, fill_value = 0)
        f_imag = scipy.interpolate.interp1d(K_x, S_Kx_Kr_mf[i,:].imag, kind = 'linear', bounds_error = 0, fill_value = 0)
        
        S[i,:] += f_real(K_xi) + 1j*f_imag(K_xi)
    
    S = np.nan_to_num(S)
    # [p1,p2] = _phs_inscribe(np.abs(S))
    # S_new = S[p1[1]:p2[1],
    #           p1[0]:p2[0]]
    S_new = S
    
    #Create window
    win_x = taylor_win(S_new.shape[1],taylor)
    win_x = np.tile(win_x, [S_new.shape[0],1])
    
    win_y = taylor_win(S_new.shape[0],taylor)
    win_y = np.array([win_y]).T
    win_y = np.tile(win_y, [1,S_new.shape[1]])
    
    win = win_x*win_y
    
    #Apply window
    S_win = S_new*win
    
    #Pad Spectrum
    length = 2**(int(np.log2(S_new.shape[0]*upsample))+1)
    pad_x = length-S_win.shape[1]
    pad_y = length-S_win.shape[0]
    S_pad = np.pad(S_win,((pad_y//2, pad_y//2),(pad_x//2,pad_x//2)), mode = 'constant')
    
    img = ift2(S_pad)
    
    return(img)
    
    
def backprojection(phs, platform, img_plane, taylor = 20, upsample = 6, prnt = False):
    """
    Backprojection Algorithm for Synthetic Aperture Radar (SAR) image formation.
    
    This function implements the Backprojection algorithm for SAR image reconstruction.
    The backprojection algorithm is a time-domain approach that coherently sums the 
    contribution from each pulse to every pixel in the output image. For each pixel,
    the algorithm calculates the range difference from each platform position, 
    interpolates the appropriate range-compressed data, and applies the proper phase 
    correction. This approach is computationally intensive but provides excellent 
    image quality and can handle arbitrary flight geometries.
    
    Parameters
    ----------
    phs : numpy.ndarray
        Complex phase history data array with shape (npulses, nsamples).
        Contains the received radar signal data for each pulse and frequency sample.
    
    platform : dict
        Platform parameters dictionary containing:
        - 'nsamples' : int, number of range samples per pulse
        - 'npulses' : int, number of radar pulses
        - 'k_r' : numpy.ndarray, radial wavenumber values
        - 'pos' : numpy.ndarray, platform positions for each pulse (3D coordinates)
        - 'delta_r' : float, range resolution
    
    img_plane : dict
        Image plane parameters dictionary containing:
        - 'u' : numpy.ndarray, range coordinates in image plane
        - 'v' : numpy.ndarray, cross-range coordinates in image plane  
        - 'pixel_locs' : numpy.ndarray, 3D pixel locations (x,y,z) for each image pixel
    
    taylor : float, optional
        Taylor window sidelobe level in dB for sidelobe suppression (default: 20).
        Applied to both range and azimuth dimensions to reduce imaging artifacts.
    
    upsample : int, optional
        Upsampling factor for zero-padding in range compression (default: 6).
        Controls the interpolation quality in the range dimension.
    
    prnt : bool, optional
        Flag to enable/disable progress printing during processing (default: True).
        When True, prints progress messages for each pulse being processed.
    
    Returns
    -------
    numpy.ndarray
        Complex-valued SAR image with shape (nv, nu) where nv and nu are the 
        dimensions of the image plane. The image represents the reconstructed 
        radar reflectivity in the spatial domain.
    
    Notes
    -----
    The Backprojection Algorithm consists of the following key steps:
    1. Apply Taylor windowing and wavenumber filtering to phase history data
    2. Zero-pad the phase history for improved range interpolation
    3. Perform range compression via Fourier transform
    4. For each pulse, calculate range differences from platform to each pixel
    5. Interpolate range-compressed data at calculated range differences
    6. Apply phase correction and coherently sum contributions from all pulses
    7. Apply final phase correction referenced to aperture center
    
    Algorithm Characteristics:
    - Exact reconstruction for arbitrary flight geometries
    - Computationally intensive (O(N³) complexity)
    - Excellent for wide-angle collections and non-linear flight paths
    - No approximations in the imaging geometry
    - Requires explicit pixel location coordinates
    
    The algorithm calculates the range difference (dr_i) between the platform position
    and each pixel for every pulse, then interpolates the range-compressed data at
    these range differences. The final image is formed by coherently summing all
    pulse contributions with proper phase corrections.
    
    References
    ----------
    Further details of this method are given in:
    - Jakowatz, C.V., et al. "Spotlight-Mode Synthetic Aperture Radar"
    - Soumekh, M. "Synthetic Aperture Radar Signal Processing"
    - Cumming, I.G. and Wong, F.H. "Digital Processing of Synthetic Aperture Radar Data"
    """
    
    #Retrieve relevent parameters
    nsamples    =   platform['nsamples']
    npulses     =   platform['npulses']
    k_r         =   platform['k_r']
    pos         =   platform['pos']
    delta_r     =   platform['delta_r']
    u           =   img_plane['u']
    v           =   img_plane['v']
    r           =   img_plane['pixel_locs']
    
    #Derive parameters
    nu = u.size
    nv = v.size
    k_c = k_r[int(nsamples/2)]
    
    #Create window
    win_x = taylor_win(nsamples,taylor)
    win_x = np.tile(win_x, [npulses,1])
    
    win_y = taylor_win(npulses,taylor)
    win_y = np.array([win_y]).T
    win_y = np.tile(win_y, [1,nsamples])
    
    win = win_x*win_y
    
    #Filter phase history    
    filt = np.abs(k_r)
    phs_filt = phs*filt*win
    
    #Zero pad phase history
    N_fft = 2**(int(np.log2(nsamples*upsample))+1)
    phs_pad = pad(phs_filt, [npulses,N_fft])
    
    #Filter phase history and perform FT w.r.t t
    Q = ft(phs_pad)    
    dr = np.linspace(-nsamples*delta_r/2, nsamples*delta_r/2, N_fft)
    
    #Perform backprojection for each pulse
    img = np.zeros(nu*nv)+0j
    for i in range(npulses):
        if prnt:
            print("Calculating backprojection for pulse %i" %i)
        r0 = np.array([pos[i]]).T
        dr_i = np.linalg.norm(r0)-np.linalg.norm(r-r0, axis = 0)
    
        Q_real = np.interp(dr_i, dr, Q[i].real)
        Q_imag = np.interp(dr_i, dr, Q[i].imag)
        
        Q_hat = Q_real+1j*Q_imag        
        img += Q_hat*np.exp(-1j*k_c*dr_i)
    
    r0 = np.array([pos[int(npulses/2)]]).T
    dr_i = np.linalg.norm(r0)-np.linalg.norm(r-r0, axis = 0)
    img = img*np.exp(1j*k_c*dr_i)   
    img = np.reshape(img, [nv, nu])[::-1,:]
    return(img)


def frequency_azimuth_to_isar_image(data, platform, size=(256, 256), window=None):
    """
    Inverse Synthetic Aperture Radar (ISAR) image formation from frequency-azimuth data.
    
    This function generates 2D ISAR images from frequency response data collected across 
    multiple azimuth angles. The algorithm assumes the target is rotating while the radar 
    is stationary (inverse synthetic aperture). The frequency-azimuth data is interpolated 
    from polar coordinates to a rectangular grid in the spatial frequency domain, then 
    transformed to produce the final ISAR image.
    
    SUGGESTED ALTERNATIVE NAME: `frequency_azimuth_to_isar_image`
    This name better describes the function's purpose: converting frequency vs azimuth 
    data into an ISAR image, which is clearer than the generic "isar_2d".
    
    Parameters
    ----------
    data : numpy.ndarray
        Complex frequency response data with shape [phi, freq] where phi represents
        azimuth angles and freq represents frequencies. The function automatically
        transposes this to the required [freq, phi] format for processing.
    
    platform : dict
        dictionary containing platform parameters, including:
        - 'freq' : numpy.ndarray, frequency values corresponding to the data
        - 'phis' : numpy.ndarray, azimuth angles corresponding to the data
    
    
    size : tuple of int, optional
        Output image size as (downrange_pixels, crossrange_pixels) (default: (256, 256)).
        The first element controls range resolution, the second controls cross-range resolution.
    
    window : str, optional
        Windowing function to apply for sidelobe suppression (default: 'flat' for no windowing).
        Other options depend on the window_function implementation.
    
    Returns
    -------
    numpy.ndarray
        2D ISAR image with specified size. The image represents the radar cross-section
        of the target in range (downrange) and cross-range (Doppler) coordinates.
        Values depend on the 'function' parameter applied to the complex image data.
    
    Notes
    -----
    Algorithm Steps:
    1. Transpose input data from [phi, freq] to [freq, phi] format
    2. Convert azimuth angles to slant plane coordinates
    3. Map frequency-azimuth data to spatial frequencies (fx, fy) in polar coordinates
    4. Interpolate polar data onto rectangular spatial frequency grid
    5. Apply windowing for sidelobe reduction
    6. Zero-pad to desired output size
    7. Apply 2D inverse FFT to obtain spatial domain image
    8. Apply specified mathematical function and transpose for output format
    
    Key Assumptions:
    - Target rotation creates synthetic aperture (ISAR geometry)
    - Frequency data represents range information via time-of-flight
    - Azimuth angles represent Doppler/cross-range information via target rotation
    - Data is collected on a planar surface (typically slant plane)
    
    Resolution Relationships:
    - Downrange resolution: Rd = c/(2 * frequency_span)
    - Cross-range resolution: Rc = lambda_center/(4 * sin(azimuth_span/2))
    where c is speed of light and lambda_center is wavelength at center frequency.
    
    Image Coordinate System:
    - Range dimension: corresponds to target extent in radar line-of-sight direction
    - Cross-range dimension: corresponds to target extent perpendicular to line-of-sight
    - Positive cross-range direction is opposite to positive azimuth progression
    
    References
    ----------
    This implementation is based on algorithms described in:
    - Moses, R.L. "Inverse Synthetic Aperture Radar Imaging"
    - Chen, C.C. and Andrews, H.C. "Target-Motion-Induced Radar Imaging"
    - Walker, J.L. "Range-Doppler Imaging of Rotating Objects"
    
    Original implementation by Randy Moses, 8/22/2002; revised 10/21/2002
    Cosmetic changes by R Kipp
    """
    
    # data is provided as [phi][freq] 2D array, but the processing below is expecting
    # [freq][phi] 2D array, so we need to transpose the data. This [phi][freq] format is the standard output
    # order from Perceive EM simulation
    data = np.transpose(data)
    c0 = 3e8
    if window is None:
        window = 'flat'
    data = data.flatten()
    # freqs = np.unique(data.index.get_level_values('Freq')) * 1.e9  # TODO make this automatic!

    freqs = platform['freq']
    
    nfreq = len(freqs)
    freqs = freqs.reshape(-1, 1)
    if 'phi' in platform.keys():
        phis = platform['phi']
    else:
        phis = platform['pos_spherical'][:,1]
    nphi = len(phis)
    phis = phis.reshape(1, -1)

    ndrng = size[0]
    nxrng = size[1]

    cel = 1
    sel = 0
    az = -phis
    az = np.unwrap(np.radians(az))
    az_ctr = np.mean(az)
    az_slant = az - az_ctr
    az_slant = az * cel
    az_slant = az_slant + az_ctr

    # generate inv synthetic aperture SAR image from frequency response vs. azimuth
    #
    # Notes:
    #  1) We assume that the frequency data is polar samples on a plane. f and
    #     az measure the range and angle on that plane.  The image is formed on
    #     the same plane.  Usually this is the slant plane.  Note for slant
    #     plane images, the azimuth angle measured on the plane is related to
    #     the azimuth angle measured from the xy-plane by:
    #         az_slant = az_ground*cos(el)
    #     where el is the elevation angle of the slant plane.
    #
    #  2) To get a ground plane image, replace x by x*cos(el).
    #
    #  3) By default we interpolate to a rectangle whose fx values range from
    #     min(f) to max(f) and whose fy values range from about the fy extent in
    #     the middle of the annulus region.
    #
    #  4) We use cubic interpolation - search for cubic to change to another type
    #
    #  5) Down range resolution Rd = c/(2 f_span), Pd = Nf*Rd
    #     Cross range resolution Rc = lambda_c/(4*sin(az_span/2)), Pc = Na*Rc
    #
    #  6) Since img comes back as (Nx x Ny) and not (Ny x Nx), to form a
    #     rectified image with x-axis along the horizontal and y-axis
    #     pointing up, use something like these commands:
    #       imagesc(y,x,amp2db(transpose(img));
    #       axis xy;
    #
    #  7) Positive cross-range points in spatial direction opposite to
    #     positive progression of az_deg
    #
    #  Randy Moses, 8/22/2002; revised 10/21/2002
    #  Cosmetic changes by R Kipp

    # We assume the az vector goes from a minimum to a maximum angle
    az_span = np.max(az) - np.min(az)

    # interpolate in the rectangular domain

    fxtrue = freqs * np.cos(az - az_ctr)  # % true fx and fy locations for this f, az grid
    fxtrue = fxtrue.reshape(-1)
    fytrue = freqs * np.sin(az - az_ctr)
    fytrue = fytrue.reshape(-1)

    # switch lower(gridsize)
    #  case {'in','inside'}
    #   %
    #   % interpolate a rectangle to the inside the populated arc
    #   %
    #   fxmin = max(fxtrue(1,:));
    #   fxmax = min(fxtrue(nf,:));
    #   fymax = max(fytrue(1,:));

    #  case {'out','outside'}
    #   %
    #   % interpolate a rectangle to the outside the populated arc
    #   %
    #   fxmin = min(fxtrue(1,:));
    #   fxmax = max(fxtrue(nf,:));
    #   fymax = max(fytrue(nf,:));

    #  case {'mid','middle'}
    #   %
    #   % interpolate a rectangle to the middle of the populated arc
    #   %
    #   fxmin = min(f_ghz);
    #   fxmax = max(f_ghz);
    #   f_c = mean(f_ghz);  % center frequency is the mean of the frequency vector
    #   fymax = sin(az_span/2)*f_c;

    # implement middle version
    fxmin = np.min(freqs)
    fxmax = np.max(freqs)
    f_c = np.mean(freqs)
    fymax = np.sin(az_span / 2) * f_c

    fx = np.linspace(fxmin, fxmax, nfreq)  # desired downrange frequencies
    fy = np.linspace(-fymax, fymax, nphi)  # desired crossrange frequencies
    grid_x, grid_y = np.meshgrid(fx, fy)
    #
    # polar to rectangular interpolation
    #
    rdata = scipy.interpolate.griddata((fxtrue, fytrue), data, (grid_x, grid_y), 'nearest', fill_value=0.)
    rdata = rdata.transpose()

    #
    # add windowing
    #
    winx, winx_sum = window_function(function=window, size=nfreq)
    winy, winy_sum = window_function(function=window, size=nphi)
    winx = winx.reshape(-1, 1)
    winy = winy.reshape(1, -1)

    # %
    # % zero padding
    # %
    if (ndrng < nfreq):
        # warning('nx should be at least as large as the length of f -- increasing nx');
        ndrng = nfreq
    if (nxrng < nphi):
        # warning('ny should be at least as large as the length of az -- increasing ny');
        nxrng = nphi

    iq = np.zeros((ndrng, nxrng), dtype=np.complex_)
    xshift = (ndrng - nfreq) // 2
    yshift = (nxrng - nphi) // 2
    iq[xshift:xshift + nfreq, yshift:yshift + nphi] = np.multiply(rdata, winx * winy)
    #
    # normalize so that unit amplitude scatterers have about unit amplitude in
    # the image (normalized for the windows).  The "about" comes because of the
    # truncation of the polar shape into a rectangular shape.
    #
    iq = np.fft.fftshift(iq) * ndrng * nxrng / winx_sum / winy_sum;
    isar_image = np.fft.fftshift(np.fft.ifft2(iq))  # Nx x Ny
    # isar_image = apply_math_function(isar_image, function)
    isar_image = isar_image.transpose()
    isar_image = isar_image[:, ::-1]
    #
    #  compute the image plane downrange and crossrange distance vectors (in
    #  meters)
    #
    dfx = fx[1] - fx[0]  # difference in x-frequencies
    dfy = fy[1] - fy[0]  # difference in y-frequencies
    dx = c0 / (2 * dfx) / ndrng
    dy = c0 / (2 * dfy) / nxrng
    x = np.transpose(np.arange(start=0, step=dx, stop=ndrng * dx))  # ndrng x 1
    y = np.arange(start=0, step=dy, stop=nxrng * dy)  # 1 x Ny
    #
    #  make the center x and y values zero
    #
    #  these two lines ensure one value of x and y are zero when nx or ny are even
    #
    range_values = x - x[ndrng // 2]
    range_values_interp = np.linspace(range_values[0], range_values[-1], num=nxrng)
    cross_range_values = y - y[nxrng // 2]
    cross_range_values_interp = np.linspace(cross_range_values[0], cross_range_values[-1], num=ndrng)
    # range_values = range_values*np.cos(az_ctr)-cross_range_values_interp*np.sin(az_ctr)
    # cross_range_values = cross_range_values*np.cos(az_ctr)+range_values_interp*np.sin(az_ctr)

    RR, XR = np.meshgrid(range_values, cross_range_values)

    indexes = zip(RR.flatten(order='F'), XR.flatten(order='F'))
    return isar_image

class SARImageProcessor:
    """
    Synthetic Aperture Radar (SAR) Image Processing Class
    
    This class provides a unified interface for multiple SAR imaging algorithms including
    Polar Format Algorithm (PFA), Omega-K, Backprojection, and ISAR processing.
    
    Example Usage:
    -------------
    # Initialize processor with data
    processor = SARImageProcessor(phase_history, platform_params)
    
    # Generate image using different algorithms
    pfa_image = processor.process_image('polar_format')
    omega_k_image = processor.process_image('omega_k')
    bp_image = processor.process_image('backprojection')
    
    # Compare algorithms
    results = processor.compare_algorithms(['polar_format', 'omega_k'])
    
    # List available methods
    methods = processor.available_methods()
    """
    
    # Class-level registry of available imaging methods
    IMAGING_METHODS = {
        'polar_format': {
            'name': 'Polar Format Algorithm',
            'description': 'Fast frequency-domain algorithm with polar-to-rectangular interpolation',
            'requires_img_plane': True,
            'output_type': 'complex',
            'complexity': 'medium',
            'best_for': 'spotlight SAR with small angles'
        },
        'omega_k': {
            'name': 'Omega-K Algorithm', 
            'description': 'Wavenumber domain algorithm with range curvature correction',
            'requires_img_plane': False,
            'output_type': 'complex',
            'complexity': 'medium',
            'best_for': 'stripmap and spotlight SAR'
        },
        'backprojection': {
            'name': 'Backprojection Algorithm',
            'description': 'Time-domain algorithm for exact reconstruction',
            'requires_img_plane': True,
            'output_type': 'complex', 
            'complexity': 'high',
            'best_for': 'wide-angle collections and arbitrary geometries'
        },
        'isar': {
            'name': 'Inverse SAR',
            'description': 'ISAR imaging from frequency-azimuth data',
            'requires_img_plane': False,
            'output_type': 'complex',
            'complexity': 'low',
            'best_for': 'rotating targets with stationary radar'
        }
    }
    
    def __init__(self, phase_history=None, platform_params=None, image_plane=None, **kwargs):
        """
        Initialize SAR Image Processor
        
        Parameters
        ----------
        phase_history : numpy.ndarray
            Complex phase history data
        platform_params : dict  
            Platform parameters dictionary
        image_plane : dict  
            image plane dictionary
        **kwargs : dict
            Additional configuration options
        """
        self.phase_history = phase_history
        self.platform_params = platform_params
        self.img_plane_params = image_plane
        self.config = kwargs
        
        # Default processing parameters
        self.default_params = {
            'taylor': 20,
            'upsample': 6,
            'res_factor': 1.0,
            'n_hat': np.array([0, 0, 1]),
            'aspect': 1,
            'print_progress': True
        }
        
        # Results cache
        self.results_cache = {}
        
    def set_data(self, phase_history, platform_params):
        """Set or update the phase history and platform parameters"""
        self.phase_history = phase_history
        self.platform_params = platform_params
        self.results_cache.clear()  # Clear cache when data changes
        
    def setup_image_plane(self, **kwargs):
        """Setup image plane parameters for algorithms that require it"""
        if self.platform_params is None:
            raise ValueError("Platform parameters must be set before setting up image plane")
            
        params = {**self.default_params, **kwargs}
        self.img_plane_params = img_plane_dict(
            self.platform_params,
            res_factor=params['res_factor'],
            n_hat=params['n_hat'],
            aspect=params['aspect'],
            upsample=params.get('img_plane_upsample', True)
        )
        return self.img_plane_params
    
    def process_image(self, method='polar_format', **kwargs):
        """
        Process SAR image using specified algorithm
        
        Parameters
        ----------
        method : str
            Imaging method: 'polar_format', 'omega_k', 'backprojection', or 'isar'
        **kwargs : dict
            Method-specific parameters
            
        Returns
        -------
        numpy.ndarray
            Processed SAR image
        """
        if method not in self.IMAGING_METHODS:
            raise ValueError(f"Unknown method '{method}'. Available: {list(self.IMAGING_METHODS.keys())}")
            
        if self.phase_history is None or self.platform_params is None:
            raise ValueError("Phase history and platform parameters must be set")
            
        # Merge default and user parameters
        params = {**self.default_params, **kwargs}
        
        # Create cache key that handles numpy arrays
        cache_key = self._create_cache_key(method, params)
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
            
        # Setup image plane if required
        method_info = self.IMAGING_METHODS[method]
        if method_info['requires_img_plane'] and self.img_plane_params is None:
            self.setup_image_plane(**params)
            
        # Process image based on method
        if method == 'polar_format':
            result = self._process_polar_format(**params)
        elif method == 'omega_k':
            result = self._process_omega_k(**params)
        elif method == 'backprojection':
            result = self._process_backprojection(**params)
        elif method == 'isar':
            result = self._process_isar(**params)
            
        # Cache result
        self.results_cache[cache_key] = result
        return result
    
    def _create_cache_key(self, method, params):
        """Create a hashable cache key from method and parameters"""
        key_parts = [method]
        
        for key, value in sorted(params.items()):
            if isinstance(value, np.ndarray):
                # Convert numpy arrays to tuples of their values
                key_parts.append((key, tuple(value.flatten())))
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to tuples
                key_parts.append((key, tuple(value)))
            else:
                # Regular hashable values
                key_parts.append((key, value))
        
        return tuple(key_parts)
    
    def _process_polar_format(self, **params):
        """Internal method for polar format processing"""
        return polar_format(
            self.phase_history,
            self.platform_params, 
            self.img_plane_params,
            taylor=params['taylor']
        )
        
    def _process_omega_k(self, **params):
        """Internal method for omega-k processing"""
        return omega_k(
            self.phase_history,
            self.platform_params,
            taylor=params['taylor'],
            upsample=params['upsample']
        )
        
    def _process_backprojection(self, **params):
        """Internal method for backprojection processing"""
        return backprojection(
            self.phase_history,
            self.platform_params,
            self.img_plane_params,
            taylor=params['taylor'],
            upsample=params['upsample'],
            prnt=params['print_progress']
        )
        
    def _process_isar(self, **params):
        """Internal method for ISAR processing"""

        return frequency_azimuth_to_isar_image(
            self.phase_history,
            self.platform_params,
            size=params.get('size', (256, 256)),
            window=params.get('window', None)
        )
    
    def compare_algorithms(self, methods=None, **kwargs):
        """
        Compare multiple imaging algorithms
        
        Parameters
        ----------
        methods : list, optional
            List of methods to compare. If None, compares all applicable methods.
        **kwargs : dict
            Parameters for processing
            
        Returns
        -------
        dict
            Dictionary with method names as keys and results as values
        """
        if methods is None:
            # Auto-select methods based on available data
            methods = ['polar_format', 'omega_k', 'backprojection']
            
        results = {}
        for method in methods:
            try:
                results[method] = self.process_image(method, **kwargs)
                print(f"✓ {self.IMAGING_METHODS[method]['name']} completed")
            except Exception as e:
                print(f"✗ {self.IMAGING_METHODS[method]['name']} failed: {e}")
                results[method] = None
                
        return results
    
    @classmethod
    def available_methods(cls):
        """Return information about available imaging methods"""
        return cls.IMAGING_METHODS.copy()
    
    @classmethod  
    def method_info(cls, method):
        """Get detailed information about a specific method"""
        if method not in cls.IMAGING_METHODS:
            raise ValueError(f"Unknown method '{method}'")
        return cls.IMAGING_METHODS[method].copy()
        
    @classmethod
    def recommend_method(cls, **criteria):
        """
        Recommend imaging method based on criteria
        
        Parameters
        ----------
        **criteria : dict
            Criteria such as 'complexity', 'accuracy', 'speed', 'geometry'
            
        Returns
        -------
        str
            Recommended method name
        """
        # Simple recommendation logic - could be expanded
        if criteria.get('geometry') == 'wide_angle':
            return 'backprojection'
        elif criteria.get('speed') == 'fast':
            return 'polar_format'
        elif criteria.get('accuracy') == 'high':
            return 'backprojection'
        else:
            return 'polar_format'  # Default recommendation
            
    def clear_cache(self):
        """Clear the results cache"""
        self.results_cache.clear()
        
    def get_algorithm_comparison_table(self):
        """Return a formatted comparison table of algorithms"""
        import pandas as pd
        
        data = []
        for method, info in self.IMAGING_METHODS.items():
            data.append({
                'Method': info['name'],
                'Output Type': info['output_type'],
                'Complexity': info['complexity'],
                'Best For': info['best_for'],
                'Requires Image Plane': info['requires_img_plane']
            })
            
        return pd.DataFrame(data)

# Example usage script for SARImageProcessor class

# Example 1: Basic usage
def example_basic_usage():
    """Basic example of using the SARImageProcessor class"""
    
    # Load your data (replace with actual data loading)
    phase_history = load_phase_history()  # Your data loading function
    platform_params = load_platform_params()  # Your platform data
    
    # Initialize processor
    processor = SARImageProcessor(phase_history, platform_params)
    
    # Process image using different algorithms
    pfa_image = processor.process_image('polar_format')
    omega_k_image = processor.process_image('omega_k')
    bp_image = processor.process_image('backprojection')
    
    return pfa_image, omega_k_image, bp_image

# Example 2: Method discovery and comparison
def example_method_discovery():
    """Example showing method discovery capabilities"""
    
    # Discover available methods
    methods = SARImageProcessor.available_methods()
    print("Available imaging methods:")
    for method_key, method_info in methods.items():
        print(f"  {method_key}: {method_info['name']}")
        print(f"    - {method_info['description']}")
        print(f"    - Best for: {method_info['best_for']}")
        print(f"    - Complexity: {method_info['complexity']}")
        print()
    
    # Get recommendation
    recommendation = SARImageProcessor.recommend_method(geometry='wide_angle')
    print(f"Recommended method for wide-angle geometry: {recommendation}")
    
    # Get algorithm comparison table
    processor = SARImageProcessor()
    comparison_table = processor.get_algorithm_comparison_table()
    print("\nAlgorithm Comparison Table:")
    print(comparison_table)

# Example 3: Algorithm comparison
def example_algorithm_comparison():
    """Example showing how to compare multiple algorithms"""
    
    # Initialize with data
    processor = SARImageProcessor(phase_history, platform_params)
    
    # Compare multiple algorithms
    methods_to_compare = ['polar_format', 'omega_k', 'backprojection']
    results = processor.compare_algorithms(methods_to_compare, taylor=25)
    
    # Process results
    for method, image in results.items():
        if image is not None:
            print(f"{method}: Image shape {image.shape}")
            # Add your visualization/analysis code here
        else:
            print(f"{method}: Failed to process")

# Example 4: Custom parameters and ISAR
def example_custom_parameters():
    """Example showing custom parameters and ISAR processing"""
    
    processor = SARImageProcessor()
    
    # Custom parameters for SAR processing
    custom_params = {
        'taylor': 25,
        'upsample': 8,
        'res_factor': 0.5,
        'print_progress': False
    }
    
    processor.set_data(phase_history, platform_params)
    image = processor.process_image('polar_format', **custom_params)
    
    # ISAR processing example
    isar_image = processor.process_image(
        'isar',
        freq_data=frequency_response_data,
        frequencies=freq_array,
        angles=azimuth_angles,
        function='abs',
        size=(512, 512)
    )
    
    return image, isar_image

if __name__ == "__main__":
    # Run examples
    example_method_discovery()
    # example_basic_usage()
    # example_algorithm_comparison()
    # example_custom_parameters()