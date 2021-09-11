D3Hit2Lead could be run on Linux and Windows.

Taking the Windows System in a laptop with only CPU as an example:

INSTALL

(1) Install Anaconda3: https://docs.anaconda.com/anaconda/install/windows/

(2) Install PyTorch: https://pytorch.org/get-started/locally/
    Run Anaconda Prompt as administrator. Then type "conda install pytorch torchvision torchaudio cpuonly -c pytorch" on the Anaconda Prompt.
    
(3) Download the whole folder of D3Hit2Lead to your local computer.

D3Hit2Lead Usage:

(1) On the Anaconda Prompt, Change directory to the D3Hit2Lead fold.

(2) Type "python  D3Hit2Lead.py" on the Anaconda Prompt.

(3) Type "1" when the script asks "Please input the number of the molecues to be modified (integer, e.g., 1):"

(4) Paste your hit smiles and protein fasta when the script asks "Please input the compound smiles and protein fasta. The smiles and fasta should be separated by space:". For example, Paste "C(/C=C(/F)\C=C(\S(=O)(=O)/C(=C/C)/C=C\CCNC(=O)/C=C/NC)/C)F MNAAAEAEFNILLATDSYKVTHYKQYPPNTSKVYSYFECREKKTENSKVRKVKYEETVFYGLQYILNKYLKGKVVTKEKIQEAKEVYREHFQDDVFNERGWNYILEKYDGHLPIEVKAVPEGSVIPRGNVLFTVENTDPECYWLTNWIETILVQSWYPITVATNSREQKKILAKYLLETSGNLDGLEYKLHDFGYRGVSSQETAGIGASAHLVNFKGTDTVAGIALIKKYYGTKDPVPGYSVPAAEHSTITAWGKDHEKDAFEHIVTQFSSVPVSVVSDSYDIYNACEKIWGEDLRHLIVSRSTEAPLIIRPDSGNPLDTVLKVLDILGKKFPVSENSKGYKLLPPYLRVIQGDGVDINTLQEIVEGMKQKKWSIENVSFGSGGALLQKLTRDLLNCSFKCSYVVTNGLGVNVFKDPVADPNKRSKKGRLSLHRTPAGTFVTLEEGKGDLEEYGHDLLHTVFKNGKVTKSYSFDEVRKNAQLNMEQDVAPH"

(5) Enjoy the predicted lead compound. The output should be like "The predicted lead (modified compound) is： Fc1cc(F)cc(c1)S(=O)(=O)c1ccc(CNC(=O)c2ccc3nccn3c2)cc1"

