# Defining an Antenna Device and Waveform

### Example
An example json file used to define the antenna device and waveform with additional comments to explain the fields

### General Format of Antenna Device File
The file is a json file with the following structure, where 3 different fields need to be defined. The waveform, post_processing and antenna fields. The waveform field defines the radar waveform parameters, the post_processing field defines the parameters for the post processing of the radar data and the antenna field defines the parameters for the antenna device.
```
"waveform":
	"mode_name": 
		waveform parameters
"post_processing":
	parametrs
"antenna": 
	"antenna_name": parameters
	"antenna_name2": parameters
```

### Example JSON FILE (not valid json file) with Comments
This file includes inline comments, although not a valid json file it can be useful to see the fields needed
```
{
	"description":"Example Antenna Device With 1 Tx and 1 Rx",
	"waveform":{
		"mode1":{
			"mode":"PulsedDoppler",			# mode can be PulseDoppler of FMCW
			"output":"RangeDoppler",		# ADC_Samples, RangeDoppler, FreqPulse
			"center_freq":77.0e9,			# Center frequency in Hz
			"bandwidth":750e6,				# Bandwidth frequency in Hz
			"num_freq_samples":700,			# Number of frequency samples for each pulse
			"cpi_duration":4.87e-3,			# length of CPI in seconds, either
			"num_pulse_CPI":200,			# <OPTIONAL> pulse_interval (s), only required if cpi_duration is not defined, pulse_interval = cpi_duration / num_pulse_CPI
			"tx_multiplex":"SIMULTANEOUS",	# SIMULTANEOUS or INDIVIDUAL. INDIVIDUAL is interleaved
			"mode_delay":"CENTER_CHIRP",	# Align timing to CENTER_CHIRP or FIRST_CHIRP
			"ADC_SampleRate":50e6, 			# ADC sampling range in HZ, required if mode is FMCW
			"isIQChannel":"True", 			# Defaults to True, if set to False, only I channel is extracted
			"tx_incident_power": 1,         # transmit power in watts, default 1
			"rx_noise_db":-120,              # noise in dB, default is not used, if parameter is included it will be added to Rx
            "rx_gain_db": 0                  # default is not used, if parameter is included it will be added to Rx
		},
	"post_processing":{
		"range_pixels": 512,				# up or downloadsampling for range output
		"doppler_pixels":256				# up or downloadsampling for doppler output
		},
	"antenna": {
		"Tx1": {
			"type":"parametric",			# type of antenna, parametric of file, if parametric beamwidth is enterer if file, ffd is defined
			"operation_mode":"tx",			# should antnena operate as tx or rx
			"polarization": "VERTICAL",		# Anteanna polarization
			"hpbwHorizDeg": 30.5,			# half power horizontal beamwidth in degrees
			"hpbwVertDeg": 60.5,			# half power vertical beamwidth in degrees
			"position":[1.15,0.0,0.5]		# Location with respect to the radar device node in meters,
			},
		"Rx1": {
			"type":"ffd",					# file mode allows antenna pattern to be defined as ffd file
			"file_path":"./beam.ffd",		# path to location of ffd, can be relative or absolute
			"operation_mode":"rx",			# should antnena operate as tx or rx
			"position":[1.5,0.0,0.5] 		# Location with respect to the radar device node in meters,
			}
	}	
}

```


### Example JSON FILE (valid json file) without Comments

```
{
	"description":"Example Antenna Device With 1 Tx and 1 Rx",
	"waveform":{
		"mode1":{
			"mode":"PulsedDoppler",
			"output":"RangeDoppler",
			"center_freq":77.0e9,
			"bandwidth":750e6,
			"num_freq_samples":700,
			"cpi_duration":4.87e-3,
			"num_pulse_CPI":200,	
			"tx_multiplex":"SIMULTANEOUS",
			"mode_delay":"CENTER_CHIRP"
			} 
		},
	"post_processing":{
		"range_pixels": 512,
		"doppler_pixels":256
		},
	"antenna": {
		"Tx1": {
			"type":"parametric",
			"operation_mode":"tx",
			"polarization": "VERTICAL",
			"hpbwHorizDeg": 30.5,
			"hpbwVertDeg": 60.5,
			"position":[1.15,0.0,0.5]
			},
		"Rx1": {
			"type":"ffd",
			"file_path":"./beam.ffd",
			"operation_mode":"rx",
			"position":[1.5,0.0,0.5]
			}
	}	
}
```

