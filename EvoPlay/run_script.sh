
conda create -n EvoPlay python=3.8 -y
source activate EvoPlay
conda update -n base conda -y
# conda install cudnn==8.2 cudatoolkit==11.3 -y 
conda install -c conda-forge openmm==7.5.1 pdbfixer matplotlib
# install alignment tools
conda install -c conda-forge -c bioconda kalign3=3.2.2 hhsuite=3.3.0 -y
conda install -y nb_conda scikit-learn biopython

pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.10+cuda11.cudnn82-cp38-none-manylinux2014_x86_64.whl
pip install jax==0.3.13

pip install seaborn logomaker tree dm-tree py3Dmol
pip install chex==0.0.7 dm-haiku==0.0.4 immutabledict==2.0.0 ml-collections==0.1.0
pip install openpyxl tape-proteins xlrd==1.2.0 


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas

#
mv Peptide_data peptide_data 