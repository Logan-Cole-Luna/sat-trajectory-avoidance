#!/bin/bash

# First install a compatible setuptools version globally
pip install setuptools==59.8.0

# Install astropy with specific options to prefer binary builds
pip install astropy==5.0.4 --only-binary :all: --no-build-isolation

# Now install the rest of the requirements
# Create a temporary requirements file without astropy
grep -v "^astropy==" requirements.txt > requirements_temp.txt
pip install -r requirements_temp.txt

# Clean up
rm requirements_temp.txt

echo "Installation completed!"
