# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based data reduction and analysis system for SANS (Small-Angle Neutron Scattering) instrument data processing. The system processes raw neutron scattering data through a multi-dimensional reduction pipeline (3D → 2D → 1D) to produce analysis-ready datasets.

## Architecture

### Core Processing Pipeline

The data reduction follows a three-stage pipeline:

1. **D3 (3D Reduction)** - `modules/D3.py`
   - Processes raw 2D detector data
   - Applies geometric corrections and efficiency calibrations
   - Outputs 2D scattering patterns

2. **D2 (2D Reduction)** - `modules/D2.py`
   - Converts 2D detector data to Q-space
   - Applies radial averaging and binning
   - Produces 2D Q-space data

3. **D1 (1D Reduction)** - `modules/D1.py`
   - Performs azimuthal averaging
   - Generates final 1D scattering curves (I vs Q)
   - Applies normalization and background subtraction

### Configuration System

- **Instrument Configuration**: `instrument_info/` directory
  - Text files defining detector geometry, distances, wavelength ranges
  - Format: key-value pairs (e.g., `L1`, `A1`, `WaveMin`, `WaveMax`)
  - Example: `instrument_info6.0-10.5A_12.75m_8mm.txt`

- **Sample Information**: `sample_info/` directory
  - Text files listing sample data files to process
  - One sample per line with metadata

- **Data Paths**: `data_dir.py`
  - Defines root data folder (currently `/data/hanzehua/vsanstrans`)
  - Update this when changing data storage location

### Key Modules

- `modules/instrument_reader.py` - Parses instrument configuration files into `instrument_info` object
- `modules/sample_reader.py` - Reads sample information files
- `modules/calculation_module.py` - Core mathematical calculations
- `modules/efficiency_calc.py` - Detector efficiency corrections
- `modules/claude_correction.py` - Correction algorithms (multiple versions)
- `modules/data_reduce_D3.py` - Alternative D3 implementation
- `path_manager.py` - Handles resource paths for PyInstaller compatibility

### Output Structure

Processing creates the following output directories:
- `Transmission/` - Transmission data
- `StitchedDataOnlySample/` - Sample-only stitched data
- `StitchedDataOnlyCell/` - Cell-only stitched data
- `StitchedDataSampleCell/` - Combined sample and cell data

## Common Commands

### Single Sample Processing

```bash
# Process samples from index START to STOP
python run.py ./instrument_info/INSTRUMENT_FILE.txt ./sample_info/SAMPLE_FILE.txt START STOP OUTPUT_PATH

# Example: Process samples 0-5
python run.py ./instrument_info/instrument_info6.0-10.5A_12.75m_8mm.txt ./sample_info/sample_info6.0-10.5A_12.75m_8mm_PS_spheres_SANS.txt 0 5 ./output_PS_spheres
```

### Batch Processing

```bash
# Parallel batch processing of multiple samples
python batchrun.py
# (Edit batchrun.py to configure sample ranges and output paths)
```

### Data Conversion

```bash
# Convert Excel file to instrument info file
python convert.py EXCEL_FILE.xlsx

# Reconvert existing data
python reconvert.py
```

### Batch Run Management

```bash
# Process multiple runs with pattern matching
python set.py
# (Edit set.py to configure batch patterns and parameters)
```

## Development Notes

### Data Flow

1. User specifies instrument config and sample info files
2. `run.py` loads configuration via `instrument_reader.py`
3. Samples are read via `sample_reader.py`
4. For each sample:
   - D3 module processes raw detector data
   - D2 module converts to Q-space
   - D1 module performs azimuthal averaging
5. Results written to output directory structure

### Configuration Parameters

Key instrument parameters in config files:
- `L1`, `L1Direct` - Primary flight path distances (mm)
- `A1`, `A1Direct`, `A2`, `A2Small` - Aperture sizes (mm²)
- `WaveMin`, `WaveMax` - Wavelength range (Å)
- `SamplePos` - Sample position (mm)
- `D1_L2`, `D2_L2`, `D3_L2` - Secondary flight paths for each detector
- `QMin`, `QMax`, `QBins` - Q-space binning parameters

### Detector Positions

- D1: 1000 mm from sample
- D2: 4000 mm from sample
- D3: 11500 mm from sample
- D4: 12820 mm from sample

### Important Implementation Details

- **Module Reloading**: `run.py` uses `importlib.reload()` to refresh D1/D2/D3 modules between samples (allows dynamic parameter updates)
- **Lambda Division**: Optional wavelength-dependent processing controlled by `instrument_info.LambdaDivided` flag
- **2D Mode**: Optional 2D-only processing controlled by `instrument_info.DataReduce2D` flag
- **Efficiency Corrections**: Applied via detector efficiency factors and time-of-flight corrections
- **Numba JIT**: D3.py uses `@jit` decorators for performance-critical calculations

### Data Dependencies

- Input data: HDF5 files (h5py) or NumPy arrays
- Intermediate: NumPy arrays in memory
- Output: NumPy files (.npy) and text files
- Plotting: Matplotlib for visualization

## Debugging Tips

- Check `data_dir.py` if data files cannot be found
- Instrument config files must exist in `instrument_info/` directory
- Sample info files must exist in `sample_info/` directory
- Output path must be writable; directories are created automatically
- Use sample index range `[0:1]` for quick single-sample testing
- Module reloading in `run.py` allows testing parameter changes without restart
