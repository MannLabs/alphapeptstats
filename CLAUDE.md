# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaStats is an open-source Python package for automated and scalable statistical analysis of mass spectrometry-based proteomics data. It supports multiple proteomics software outputs (MaxQuant, DIANN, FragPipe, Spectronaut, etc.) and provides both a Python API and a Streamlit-based GUI.

## Development Commands

### Testing
```bash
pytest                    # Run all tests
pytest tests/test_*.py    # Run specific test file
pytest --cov             # Run tests with coverage
```

### Code Quality
```bash
ruff check                        # Lint code (standard rules)
ruff check --config=ruff-lint-strict.toml  # Lint with strict rules
ruff format                       # Format code
pre-commit run --all-files        # Run all pre-commit hooks
```

### Development Setup
```bash
pip install -r requirements_dev.txt    # Install development dependencies
pre-commit install                     # Install pre-commit hooks
```

### GUI Development
```bash
alphastats gui           # Launch Streamlit GUI
streamlit run alphastats/gui/gui.py    # Alternative GUI launch
```

## Architecture Overview

### Core Components

**DataSet Class**: Central orchestrating object that provides unified interface to all functionality. It manages three core data structures:
- `rawinput`: Original loaded data
- `mat`: Processed intensity matrix (samples × features)
- `metadata`: Sample annotations and experimental design

**Data Loader System**: Plugin-based architecture with `BaseLoader` abstract base class supporting:
- MaxQuant (`maxquant_loader.py`)
- DIANN (`diann_loader.py`)
- FragPipe (`fragpipe_loader.py`)
- Spectronaut (`spectronaut_loader.py`)
- mzTab (`mztab_loader.py`)
- Generic formats (`generic_loader.py`)

**Processing Pipeline**: Modular stages with clear separation:
1. **Loading**: Raw data ingestion via format-specific loaders
2. **Harmonization**: Column mapping and data standardization
3. **Preprocessing**: Contamination removal, log transformation, normalization
4. **Analysis**: Statistical tests, differential expression, pathway analysis
5. **Visualization**: Plots and interactive graphics

### GUI Architecture (Streamlit)

**Page-Based Navigation**: Multi-page app structure in `gui/pages_/`:
- `01_Home.py`: Landing page and project overview
- `02_Import Data.py`: Data loading interface
- `03_Data Overview.py`: Data exploration and QC
- `04_Preprocessing.py`: Data cleaning and normalization
- `05_Analysis.py`: Statistical analysis interface
- `07_Results.py`: Results visualization and export

**State Management**: Sophisticated session state handling via:
- `StateKeys` class: Constants for accessing the streamlit session state
- `session_manager.py`: Session persistence and restoration
- Widget syncing between different components

**Helper Utilities** organized by responsibility:
- `analysis_helper.py`: Statistical analysis workflows
- `preprocessing_helper.py`: Data cleaning operations
- `import_helper.py`: Data loading assistance
- `ui_helper.py`: Common UI components and styling

### Key Design Patterns

- **Factory Pattern**: `factory.py` creates DataSet objects from various inputs
- **Facade Pattern**: DataSet provides simplified interface to complex subsystems
- **Strategy Pattern**: Different loaders handle various data formats
- **Constants Classes**: Type-safe string constants via `ConstantsClass` metaclass

## Important Implementation Details

### Data Flow
1. Raw data loaded via format-specific loaders
2. Data harmonized using column mapping (`harmonizer.py`)
3. Matrix creation with feature/sample validation
4. Preprocessing applied based on workflow configuration
5. Statistical analysis performed on processed data
6. Results visualized through plotting functions

### Testing Strategy
- Unit tests for core functionality in `tests/`
- GUI tests using Streamlit testing framework
- Integration tests with real proteomics datasets in `testfiles/`
- Notebook-based end-to-end testing

### Code Organization
- Core library is GUI-agnostic (can be used programmatically)
- Clear separation between data processing and visualization
- Modular design allows easy extension with new loaders/analysis methods
- Constants defined in dedicated modules for type safety

### Dependencies
- Scientific computing: NumPy, Pandas, SciPy, scikit-learn
- Visualization: Plotly (interactive), Matplotlib/Seaborn (static)
- GUI: Streamlit with custom components
- Statistics: Pingouin, statsmodels for advanced analyses
- Data formats: PyTeomics for proteomics-specific formats

## Development Guidelines

### Adding New Data Loaders
1. Inherit from `BaseLoader` in `loader/base_loader.py`
2. Implement required abstract methods (`load_file`, `load_metadata`)
3. Add format detection logic and column mapping
4. Include comprehensive tests with sample data

### Adding Statistical Methods
1. Add analysis function to appropriate module in `statistics/`
2. Update `DataSet` class with new method
3. Add GUI interface in relevant page
4. Include visualization function in `plots/`

### GUI Development
- Follow page-based architecture for new features
- Use state management patterns consistently
- Implement proper error handling and user feedback
- Test both programmatic and GUI workflows

### Testing Requirements
- Write unit tests for new functionality using pytest, parametrize for multiple cases
- Only when asked: Include integration tests with sample data
- Only when asked: Test both API and GUI interfaces
- Ensure backward compatibility with existing datasets
