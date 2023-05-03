echo "Generating matrices and measuring performance"
python3 scripts/matrices_generator.py
python3 scripts/graphs_data_generator.py
echo ""

echo "Creating archive with results and clearing"
tar -zcvf parallelrank_perfs.tar.gz parallelrank_perfs/
rm -rf parallelrank_perfs
rm -rf parallelrank_data
