echo "Installing numpy and parallelrank"
python -m pip install numpy
cd src
python setup.py install
cd ../
echo ""

echo "Generating matrices and measuring performance"
python scripts/matrices_generator.py
python scripts/graphs_data_generator.py
echo ""

echo "Creating archive with results and clearing results"
apt install zip -y
zip -r parallelrank_perfs.zip parallelrank_perfs/
rm -rf parallelrank_perfs
rm -rf parallelrank_data
