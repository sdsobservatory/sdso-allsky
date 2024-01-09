from abc import ABC, abstractmethod


class Camera(ABC):

    @property
    def shift(self):
        """
        Number of bits to shift right to scale ADU to native ADC bit depth.
        """
        return 16 - self.adc_bit_depth

    @property
    @abstractmethod
    def adc_bit_depth(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def offset(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def exposure_min(self) -> int:
        """
        Exposure min in microseconds.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def exposure_max(self) -> int:
        """
        Exposure max in microseconds.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def gain_day(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def gain_night(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_bias_median(self, exposure_us: int, gain: int) -> int:
        """
        Get the bias median for the given exposure and gain, scaled as a 16-bit integer.
        """
        raise NotImplementedError

    @abstractmethod
    def get_e_per_adu(self, gain: int) -> float:
        """
        Get the electrons per ADU for the given gain.
        Typically, this is an equation derived from the manufacturer's published e/ADU curve.
        """
        raise NotImplementedError
