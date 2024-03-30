from prometheus_client import Gauge


TIMELAPSE_EXPOSURE_METRIC = Gauge('allsky_timelapse_exposure',
                                  'Exposure of the most recent image, in seconds.')
TIMELAPSE_GAIN_METRIC = Gauge('allsky_timelapse_gain',
                              'Gain of the most recent image.')
TIMELAPSE_EXPOSURE_NEXT_METRIC = Gauge('allsky_timelapse_exposure_next',
                                       'Exposure of the next image, in seconds.')
TIMELAPSE_GAIN_NEXT_METRIC = Gauge('allsky_timelapse_gain_next',
                                   'Gain of the next image.')
TIMELAPSE_E_PER_SEC_METRIC = Gauge('allsky_timelapse_e_per_sec',
                                   'Electrons per second of the most recent image, in electrons.')
TIMELAPSE_E_PER_SEC_NEXT_METRIC = Gauge('allsky_timelapse_e_per_sec_next',
                                        'Electrons per second of the next image, in electrons.')
