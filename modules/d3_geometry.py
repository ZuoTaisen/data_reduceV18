# -*- coding: utf-8 -*-
"""
D3 Detector Geometry Configuration.
Handles all geometric calculations and detector parameter setup for D3.

Created: 2026-04-04
Author: Refactored by Claude Code
"""

import numpy as np
from d3_utils import falling_distance


class D3Geometry:
    """
    Manages D3 detector geometry and coordinate systems.

    This class encapsulates all geometric parameters and calculations
    for the D3 detector, including detector positions, pixel arrays,
    and Q-space coordinate transformations.
    """

    def __init__(self, instrument_info):
        """
        Initialize D3 geometry from instrument configuration.

        Parameters
        ----------
        instrument_info : object
            Instrument configuration object containing all detector parameters
        """
        self.info = instrument_info
        self.const = 3956.2  # Conversion constant for wavelength calculations

        # Initialize geometry parameters
        self._init_basic_geometry()
        self._init_detector_parameters()
        self._init_data_parameters()
        self._init_arrays()
        self._init_q_space()

    def _init_basic_geometry(self):
        """Initialize basic geometric parameters."""
        self.L1 = self.info.L1 + self.info.SampleDisplace  # Primary flight path (mm)
        self.L2 = self.info.D3_L2 * self.info.L2_factor - self.info.SampleDisplace  # Secondary flight path (mm)
        self.L1Direct = self.info.L1Direct  # Direct beam L1 (mm)
        self.SamplePos = self.info.SamplePos + self.info.SampleDisplace  # Sample position (mm)
        self.ModToDetector = self.SamplePos + self.L2  # Total flight path (mm)
        self.L = self.SamplePos + self.L2

        # Aperture sizes
        self.A1 = self.info.A1  # mm
        self.A2 = self.info.A2  # mm
        self.A2Small = self.info.A2Small  # mm

    def _init_detector_parameters(self):
        """Initialize detector physical parameters."""
        self.DetFactor = self.info.DetFactor  # Detector size scaling factor
        self.TubeWidth = 8.5  # mm
        self.TubeHeight = 4 * self.DetFactor  # mm
        self.BankGap = 6.4  # mm between banks
        self.ModuleGap = 2  # mm between modules
        self.ModuleWidth = self.TubeWidth * 16
        self.BankWidth = self.TubeWidth * 64 * 2 + self.ModuleGap * 6 + self.BankGap  # mm
        self.BankHeight = 1000 * self.DetFactor - self.TubeHeight  # mm

        # Detector binning
        self.XBins = 128
        self.YBins = 250

    def _init_data_parameters(self):
        """Initialize data acquisition and binning parameters."""
        self.WaveMin = self.info.WaveMin  # Angstroms
        self.WaveMax = self.info.WaveMax  # Angstroms
        self.TimeDelay = self.info.TimeDelayD3  # ms
        self.TOF = self.info.TOF  # Time-of-flight window (ms)
        self.WaveBins = 250
        self.WaveBinsSelected = self.info.WaveBinsSelectedD3

        # Q-space parameters
        self.QMin = self.info.QMin  # A^-1
        self.QMax = self.info.QMax  # A^-1
        self.QBins = self.info.QBins
        self.QMax2D = self.info.QMax2D
        self.QBins2D = int(self.QMax2D / 0.12 * 250)

        # Radial binning
        self.RMin = 1E-3  # mm - small positive number to avoid division by zero
        self.R = 900  # mm - maximum radius
        self.RBins = 200

        # Beam stop
        self.BeamStopDia = self.info.BeamStopDia

        # Resolution parameters
        self.DeltaLambdaRatio = 0.019 * self.const / self.ModToDetector
        self.SampleThickness = 1  # mm

    def _init_arrays(self):
        """Initialize coordinate arrays and centers."""
        # Beam center
        self.XCenter0 = self.info.XCenter
        self.YCenter0 = self.info.YCenter

        # Radial array with logarithmic spacing
        self.RArrayEdges = np.logspace(np.log10(self.RMin), np.log10(self.R), self.RBins + 1)
        self.RArray = np.sqrt(self.RArrayEdges[:-1] * self.RArrayEdges[1:])  # Geometric mean for bin centers

        # Calculate L2 and total path for each radius
        self.L2_Array = np.sqrt(self.L2**2 + self.RArray**2)
        self.L_Array = self.SamplePos + self.L2_Array

        # Beam stop start bin
        self.BeamStopStart = np.digitize(self.BeamStopDia / 2, self.RArrayEdges)

        # Angular arrays
        self.ThetaArray = np.arctan(self.RArray / self.L2_Array)

        # Time-of-flight array
        self.TofArray = np.linspace(self.TimeDelay, self.TimeDelay + self.TOF, self.WaveBins) + \
                        self.TOF / self.WaveBins / 2

        # Wavelength arrays
        self.StartWave = self.info.StartWave
        self.StopWave = self.info.StopWave

        # TOF bounds for each radial bin (position-dependent)
        self.TofMin = self.L_Array[:, None] * self.StartWave / self.const  # ms
        self.TofMax = self.L_Array[:, None] * self.StopWave / self.const  # ms

        # Wavelength selection mask (RBins × WaveBins)
        self.WavelengthArrayBool = (self.TofArray[None, :] > self.TofMin) * \
                                   (self.TofArray[None, :] < self.TofMax)

        # Wavelength arrays (position-dependent and single position)
        self.WavelengthArray2 = self.const * self.TofArray / self.L_Array[:, None]
        self.WavelengthArray = self.const * self.TofArray / self.L

        # Detector pixel arrays
        self._init_pixel_arrays()

        # Gravity correction
        self.FallingDistanceDirect = falling_distance(self.WavelengthArray, self.L1Direct, self.L2)
        self.FallingDistance = falling_distance(self.WavelengthArray, self.L1, self.L2)

        # Cosine correction factor
        self.CosAlpha = self.L2 / np.sqrt(self.YArray**2 + self.L2**2)

    def _init_pixel_arrays(self):
        """Initialize X and Y pixel position arrays."""
        # Module arrays
        self.ModuleArray = np.arange(0, self.TubeWidth * 16, self.TubeWidth)
        self.FirstArray = self.ModuleArray - self.BankWidth / 2 + self.TubeWidth / 2
        self.ArrayDistance = self.ModuleWidth + self.ModuleGap

        # X arrays for left and right banks
        self.XArrayLeft = np.concatenate((
            self.FirstArray,
            self.FirstArray + self.ArrayDistance,
            self.FirstArray + self.ArrayDistance * 2,
            self.FirstArray + self.ArrayDistance * 3
        ))
        self.XArrayRight = np.concatenate((
            self.FirstArray + self.ArrayDistance * 4 + self.BankGap,
            self.FirstArray + self.ArrayDistance * 5 + self.BankGap,
            self.FirstArray + self.ArrayDistance * 6 + self.BankGap,
            self.FirstArray + self.ArrayDistance * 7 + self.BankGap
        )) - 2

        self.XArray0 = np.concatenate((self.XArrayLeft, self.XArrayRight))

        # Y array
        self.YArray = np.arange(
            -1 * self.TubeHeight * self.YBins / 2 + self.TubeHeight / 2,
            self.TubeHeight * self.YBins / 2 + self.TubeHeight / 2,
            self.TubeHeight
        )

        # Calculate actual beam center positions
        self.XCenter = self.XArray0[int(self.XCenter0)] * (1 - self.XCenter0 % 1) + \
                       self.XArray0[int(self.XCenter0) + 1] * (self.XCenter0 % 1)
        self.YCenter = self.YArray[int(self.YCenter0)] * (1 - self.YCenter0 % 1) + \
                       self.YArray[int(self.YCenter0) + 1] * (self.YCenter0 % 1)

        # Shift arrays to centered coordinates
        self.XArray = self.XArray0 - self.XCenter
        self.YArray = self.YArray + self.YCenter

        # Create shifted arrays for sub-pixel binning
        self.XArrayLeft_shifted = self._shift_array_left(self.XArray)
        self.XArrayRight_shifted = self._shift_array_right(self.XArray)

    def _shift_array_left(self, arr):
        """Shift array elements left by 1/3 of spacing."""
        result = np.zeros_like(arr)
        result[1:] = arr[1:] - (arr[1:] - arr[:-1]) / 3
        result[0] = arr[0] - (arr[1] - arr[0]) / 3
        return result

    def _shift_array_right(self, arr):
        """Shift array elements right by 1/3 of spacing."""
        result = np.zeros_like(arr)
        result[:-1] = arr[:-1] + (arr[1:] - arr[:-1]) / 3
        result[-1] = arr[-1] + (arr[-1] - arr[-2]) / 3
        return result

    def _init_q_space(self):
        """Initialize Q-space coordinate arrays."""
        # 1D Q arrays
        self.QX = self.info.QX
        self.QY = np.zeros(len(self.QX))

        # 2D Q arrays
        self.QX2D = np.linspace(-1 * self.QMax2D, self.QMax2D, self.QBins2D)
        self.QY2D = np.linspace(-1 * self.QMax2D, self.QMax2D, self.QBins2D)
        self.QArray2d = np.zeros((len(self.QX2D), len(self.QY2D)))

        # D3-specific Q arrays
        self.D3ThetaX = np.arctan(self.XArray / self.L2)
        self.D3ThetaXLeft = np.arctan(self.XArrayLeft_shifted / self.L2)
        self.D3ThetaXRight = np.arctan(self.XArrayRight_shifted / self.L2)
        self.D3ThetaY = np.arctan((self.YArray[:, None] - self.FallingDistance) / self.L2)

        # Q-space coordinates
        self.D3QX = 4 * np.pi * np.sin(self.D3ThetaX[:, None] / 2) / self.WavelengthArray
        self.D3QXLeft = 4 * np.pi * np.sin(self.D3ThetaXLeft[:, None] / 2) / self.WavelengthArray
        self.D3QXRight = 4 * np.pi * np.sin(self.D3ThetaXRight[:, None] / 2) / self.WavelengthArray
        self.D3QY = 4 * np.pi * np.sin(self.D3ThetaY / 2) / self.WavelengthArray

    def delta_q_calc(self, wavelength, r):
        """
        Calculate Q resolution function.

        Parameters
        ----------
        wavelength : array_like
            Wavelength array (Angstroms)
        r : array_like
            Radius array (mm)

        Returns
        -------
        array_like
            2D array of delta-Q values (resolution)
        """
        LP = 1 / (1 / self.L1 + 1 / self.L2)
        delta_q = (1 / 12) * (2 * np.pi / wavelength) * (
            3 * (self.A1 / 2)**2 / self.L1**2 +
            3 * (self.A2 / 2)**2 / LP**2 +
            (self.TubeWidth**2 + self.TubeHeight**2) / self.L2**2 +
            r[:, None]**2 / self.L2**2 * (self.DeltaLambdaRatio)**2
        )
        return delta_q

    def q_calc(self, wavelength, r):
        """
        Calculate Q array from wavelength and radius.

        Parameters
        ----------
        wavelength : array_like
            Wavelength array (Angstroms)
        r : array_like
            Radius array (mm)

        Returns
        -------
        array_like
            2D Q array (A^-1)
        """
        theta = np.arctan(r[:, None] / self.L2)
        q = 4 * np.pi * np.sin(theta / 2) / wavelength
        return q
