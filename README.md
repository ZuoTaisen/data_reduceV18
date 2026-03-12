# Data Reduce V18

A comprehensive Python-based data reduction and analysis system for Small-Angle Neutron Scattering (SANS) instrument data processing.

## Overview

This project processes raw neutron scattering data from SANS instruments through a sophisticated multi-dimensional reduction pipeline. The system converts raw detector counts into analysis-ready scattering curves by applying geometric corrections, efficiency calibrations, and Q-space transformations.

### What is SANS?

Small-Angle Neutron Scattering (SANS) is a powerful technique for studying the structure of materials at length scales of 1-100 nm. Neutrons scatter off atomic nuclei, providing complementary information to X-ray scattering. This project processes the raw scattering data from SANS instruments into meaningful scientific results.

## Key Features

- **Multi-Dimensional Reduction Pipeline**: Processes data through 3D → 2D → 1D reduction stages
- **Flexible Configuration**: Instrument-specific parameters via configuration files
- **Batch Processing**: Parallel processing of multiple samples for high-throughput analysis
- **Efficiency Corrections**: Detector efficiency calibrations and time-of-flight corrections
- **Multiple Output Formats**: Transmission data, stitched data (sample-only, cell-only, combined)
- **Data Conversion Tools**: Excel to instrument configuration file conversion
- **Visualization Support**: Built-in plotting utilities for 2D and 1D data

## Project Structure

```
data_reduceV18/
├── run.py                      # Main processing script
├── batchrun.py                 # Parallel batch processing
├── convert.py                  # Excel to instrument info conversion
├── reconvert.py                # Data reconversion utility
├── set.py                      # Batch run manager
├── data_dir.py                 # Data path configuration
├── path_manager.py             # Resource path management
│
├── modules/                    # Core processing modules
│   ├── D1.py                   # 1D reduction (azimuthal averaging)
│   ├── D2.py                   # 2D reduction (Q-space conversion)
│   ├── D3.py                   # 3D reduction (raw detector processing)
│   ├── instrument_reader.py    # Instrument configuration parser
│   ├── sample_reader.py        # Sample information reader
│   ├── calculation_module.py   # Core calculations
│   ├── efficiency_calc.py      # Detector efficiency corrections
│   └── claude_correction.py    # Correction algorithms
│
├── instrument_info/            # Instrument configuration files
├── sample_info/                # Sample information files
├── masks/                      # Detector mask utilities
├── plot2D/                     # 2D plotting and visualization
├── npyfiles/                   # NumPy data storage
└── CLAUDE.md                   # Developer documentation
```

## Installation

### Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- h5py
- openpyxl
- numba

### Setup

```bash
# Clone the repository
git clone https://github.com/ZuoTaisen/data_reduceV18.git
cd data_reduceV18

# Install dependencies
pip install numpy scipy matplotlib h5py openpyxl numba
```

## Usage

### Basic Single Sample Processing

```bash
python run.py <instrument_config> <sample_info> <start_index> <stop_index> <output_path>
```

**Example:**
```bash
python run.py ./instrument_info/instrument_info6.0-10.5A_12.75m_8mm.txt \
              ./sample_info/sample_info6.0-10.5A_12.75m_8mm_PS_spheres_SANS.txt \
              0 5 ./output_PS_spheres
```

This processes samples 0-5 and saves results to `./output_PS_spheres/`

### Batch Processing

For processing multiple samples in parallel:

```bash
python batchrun.py
```

Edit `batchrun.py` to configure:
- Sample ranges
- Number of parallel workers
- Output paths

### Data Conversion

Convert Excel files to instrument configuration format:

```bash
python convert.py <excel_file>
```

## Configuration

### Instrument Configuration Files

Located in `instrument_info/`, these text files define detector geometry and instrument parameters:

```
L1: 12750          # Primary flight path (mm)
A1: 50             # Primary aperture (mm²)
A2: 25             # Secondary aperture (mm²)
WaveMin: 6.0       # Minimum wavelength (Å)
WaveMax: 10.5      # Maximum wavelength (Å)
SamplePos: 1000    # Sample position (mm)
```

### Sample Information Files

Located in `sample_info/`, these files list data files to process:

```
sample_name_1.h5
sample_name_2.h5
sample_name_3.h5
```

### Data Path Configuration

Edit `data_dir.py` to set the root data directory:

```python
DataFold = r'/path/to/your/data'
```

## Output Structure

Processing generates the following output directories:

```
output_path/
├── Transmission/                    # Transmission data
├── StitchedDataOnlySample/         # Sample-only stitched data
├── StitchedDataOnlyCell/           # Cell-only stitched data
└── StitchedDataSampleCell/         # Combined sample and cell data
```

Each contains:
- 1D scattering curves (I vs Q)
- 2D Q-space maps
- Metadata and processing logs

## Processing Pipeline

### Stage 1: D3 (3D Reduction)
- Processes raw 2D detector data
- Applies geometric corrections
- Applies detector efficiency calibrations
- Outputs 2D scattering patterns

### Stage 2: D2 (2D Reduction)
- Converts 2D detector data to Q-space
- Applies radial averaging and binning
- Produces 2D Q-space data

### Stage 3: D1 (1D Reduction)
- Performs azimuthal averaging
- Generates final 1D scattering curves
- Applies normalization and background subtraction

## Advanced Features

### Lambda Division

Process data separately for different wavelength ranges:

```python
instrument_info.LambdaDivided = True
```

### 2D-Only Processing

Skip 1D reduction and output only 2D data:

```python
instrument_info.DataReduce2D = True
```

### Efficiency Corrections

Detector efficiency factors are applied automatically based on:
- Detector response calibration
- Time-of-flight corrections
- Wavelength-dependent efficiency

## Performance Optimization

- **Numba JIT Compilation**: Performance-critical calculations use Numba for speed
- **Parallel Batch Processing**: Use `batchrun.py` for multi-core processing
- **Module Reloading**: Dynamic parameter updates without restart

## Troubleshooting

- **Data not found**: Check `data_dir.py` and ensure data paths are correct
- **Configuration errors**: Verify instrument config files exist in `instrument_info/`
- **Output permission denied**: Ensure output directory is writable
- **Memory issues**: Process smaller sample ranges or reduce Q-space binning

## Developer Documentation

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation and development guidelines.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- Changes are well-documented
- New features include appropriate corrections/calibrations

## License

This project is maintained by the SANS data analysis team.

## Contact

For questions or issues, please contact the project maintainers.

## References

- SANS technique overview: https://www.nist.gov/ncnr/sans
- Neutron scattering data reduction: https://www.ornl.gov/
- Q-space analysis: https://en.wikipedia.org/wiki/Small-angle_scattering
